"""
Модули слияния модальностей.

Реализуем ДВА варианта, чтобы потом сравнить (это будет важная аблация в
курсовой):

1) ConcatFusion — простое решение:
   - интерполируем видео-фичи до длины аудио (или наоборот);
   - конкатенируем по фичевой оси: (B, T, d_a + d_v);
   - линейная проекция обратно в d_model.

2) CrossAttentionFusion — основной вариант:
   - аудио-фичи (Q) делают attention к видео-фичам (K, V)  → аудио, обогащённое видео;
   - видео-фичи (Q) делают attention к аудио-фичам (K, V)  → видео, обогащённое аудио;
   - 2 таких блока подряд, в конце — конкатенация и проекция.

Зачем оба: показать в эксперименте, что cross-attention лучше при шуме.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_key_padding_mask(lens: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Создаёт булеву маску паддинга для MultiheadAttention.
    mask[b, t] = True означает «этот шаг — паддинг, игнорировать».
    """
    arange = torch.arange(max_len, device=lens.device).unsqueeze(0)  # (1, T)
    return arange >= lens.unsqueeze(1)                                 # (B, T)


def _interpolate_to(
    x: torch.Tensor, target_len: int
) -> torch.Tensor:
    """
    Временная интерполяция (B, T_src, D) → (B, target_len, D).
    Использует linear interpolation по временной оси.
    """
    if x.shape[1] == target_len:
        return x
    # F.interpolate работает с (B, C, L) — транспонируем
    x = x.transpose(1, 2)            # (B, D, T)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return x.transpose(1, 2)         # (B, target_len, D)


# =============================================================================
# ConcatFusion
# =============================================================================

class ConcatFusion(nn.Module):
    """
    Простое слияние: интерполируем видео до длины аудио → конкат → проекция.

    Args:
        d_audio: размер аудио-эмбеддингов.
        d_video: размер видео-эмбеддингов.
        d_model: размер выходного пространства.
    """

    def __init__(self, d_audio: int, d_video: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_audio + d_video, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        video: torch.Tensor,
        video_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio:      (B, Ta, d_audio)
            audio_lens: (B,)
            video:      (B, Tv, d_video)
            video_lens: (B,)  — не используются напрямую, длина наследуется от аудио

        Returns:
            fused: (B, Ta, d_model)
            lens:  (B,)  — audio_lens
        """
        Ta = audio.shape[1]
        # Выравниваем видео по временной оси аудио
        video_interp = _interpolate_to(video, Ta)  # (B, Ta, d_video)
        fused = self.proj(torch.cat([audio, video_interp], dim=-1))
        return fused, audio_lens


# =============================================================================
# CrossAttentionFusion
# =============================================================================

class _CrossAttentionBlock(nn.Module):
    """
    Один блок двунаправленного cross-attention:
      a' = LayerNorm(a + MHA(q=a, k=v, v=v))
      a' = LayerNorm(a' + FFN(a'))
      v' = LayerNorm(v + MHA(q=v, k=a, v=a))
      v' = LayerNorm(v' + FFN(v'))

    Args:
        d_model: единый размер пространства признаков (аудио и видео уже спроецированы).
        n_heads: число голов attention.
        dropout: dropout.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        ffn_dim = d_model * 4

        # Audio ← Video
        self.a2v_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.a_norm1 = nn.LayerNorm(d_model)
        self.a_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.a_norm2 = nn.LayerNorm(d_model)

        # Video ← Audio
        self.v2a_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.v_norm1 = nn.LayerNorm(d_model)
        self.v_ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.v_norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        video: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio:      (B, Ta, d_model)
            audio_mask: (B, Ta)  — True = паддинг
            video:      (B, Tv, d_model)
            video_mask: (B, Tv)  — True = паддинг

        Returns:
            audio': (B, Ta, d_model)
            video': (B, Tv, d_model)
        """
        # --- Audio attends to Video ---
        # pre-norm: нормируем перед attention (более стабильно)
        a_q = self.a_norm1(audio)
        a_attn, _ = self.a2v_attn(
            query=a_q,
            key=video,
            value=video,
            key_padding_mask=video_mask,
        )
        audio = audio + a_attn
        audio = audio + self.a_ffn(self.a_norm2(audio))

        # --- Video attends to Audio ---
        v_q = self.v_norm1(video)
        v_attn, _ = self.v2a_attn(
            query=v_q,
            key=audio,
            value=audio,
            key_padding_mask=audio_mask,
        )
        video = video + v_attn
        video = video + self.v_ffn(self.v_norm2(video))

        return audio, video


class CrossAttentionFusion(nn.Module):
    """
    Слияние через перекрёстное внимание (cross-attention).

    Шаги:
      1) Проецируем аудио (d_audio → d_model) и видео (d_video → d_model),
         если размеры отличаются.
      2) Прогоняем через n_layers блоков двунаправленного cross-attention.
      3) Интерполируем видео до длины аудио, конкатенируем, проецируем назад.

    Args:
        d_audio:  размер аудио-эмбеддингов.
        d_video:  размер видео-эмбеддингов.
        d_model:  единый внутренний размер (и размер выхода).
        n_heads:  число голов attention.
        n_layers: число блоков cross-attention.
        dropout:  dropout в attention и FFN.
    """

    def __init__(
        self,
        d_audio: int,
        d_video: int,
        d_model: int,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Входные проекции (могут быть identity если d == d_model)
        self.audio_proj = (
            nn.Linear(d_audio, d_model) if d_audio != d_model else nn.Identity()
        )
        self.video_proj = (
            nn.Linear(d_video, d_model) if d_video != d_model else nn.Identity()
        )

        self.layers = nn.ModuleList(
            [_CrossAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Финальная проекция: конкат(audio', video') → d_model
        self.out_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        video: torch.Tensor,
        video_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio:      (B, Ta, d_audio)
            audio_lens: (B,)
            video:      (B, Tv, d_video)
            video_lens: (B,)

        Returns:
            fused: (B, Ta, d_model)
            lens:  (B,)  — audio_lens (выходная длина = аудио-длина)
        """
        Ta = audio.shape[1]
        Tv = video.shape[1]

        # Проецируем в единое пространство
        audio = self.audio_proj(audio)  # (B, Ta, d_model)
        video = self.video_proj(video)  # (B, Tv, d_model)

        # Маски паддинга
        audio_mask = _make_key_padding_mask(audio_lens, Ta)
        video_mask = _make_key_padding_mask(video_lens, Tv)

        # Прогоняем через блоки cross-attention
        for layer in self.layers:
            audio, video = layer(audio, audio_mask, video, video_mask)

        # Выравниваем видео по временной оси аудио и сливаем
        video_aligned = _interpolate_to(video, Ta)
        fused = self.out_proj(torch.cat([audio, video_aligned], dim=-1))

        return fused, audio_lens
