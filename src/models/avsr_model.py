"""
Главная модель AVSR — собирает все компоненты воедино.

Архитектура:
  audio_mel ──► AudioEncoder ──┐
                                ├──► Fusion ──► CTC Head ──► logits
  lip_video ──► VideoEncoder ──┘

Поддерживает три режима:
  - "audio_only":  игнорирует видео (для бейзлайна)
  - "video_only":  игнорирует аудио (бейзлайн чтения по губам)
  - "av":          мультимодальный (основной)

Modality dropout (важная штука): при обучении с вероятностью p_drop
случайно зануляем одну из веток. Это учит модель не разваливаться,
если в инференсе одна модальность пропала (например, лицо не нашлось).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.audio_encoder import AudioEncoder
from src.models.video_encoder import VideoEncoder
from src.models.fusion import ConcatFusion, CrossAttentionFusion


class AVSRModel(nn.Module):
    """
    Полная мультимодальная AVSR модель.

    Конструируется из OmegaConf конфига (cfg). Поддерживает:
      - mode "av":         аудио + видео (основной)
      - mode "audio_only": только аудио (бейзлайн)
      - mode "video_only": только видео / чтение по губам (бейзлайн)
      - fusion_type "cross_attention" (по умолчанию) или "concat"
      - modality_dropout: вероятность случайного обнуления одной из модальностей
        во время обучения (делает модель устойчивой к отсутствию одной модальности)

    Параметры конфига (cfg.model.*):
        mode:              "av" / "audio_only" / "video_only"
        d_model:           int, размер скрытого пространства (для fusion и выхода)
        modality_dropout:  float, 0.0 = выключен
        vocab_size:        int, размер алфавита (включая CTC blank)

        audio_encoder.name:   str, HuggingFace model id
        audio_encoder.freeze: bool

        video_encoder.d_video: int
        video_encoder.n_layers: int
        video_encoder.n_heads:  int

        fusion.type:     "cross_attention" / "concat"
        fusion.n_layers: int
        fusion.n_heads:  int
        fusion.dropout:  float
    """

    VALID_MODES = {"av", "audio_only", "video_only"}

    def __init__(self, cfg, vocab_size: int) -> None:
        super().__init__()

        mcfg = cfg.model
        mode = mcfg.get("mode", "av")
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode должен быть одним из {self.VALID_MODES}, получили '{mode}'")
        self.mode = mode
        self.modality_dropout: float = float(mcfg.get("modality_dropout", 0.0))

        d_model: int = int(mcfg.d_model)

        # ── Аудио-энкодер ────────────────────────────────────────────────────
        acfg = mcfg.audio_encoder
        self.audio_encoder = AudioEncoder(
            model_name=str(acfg.get("name", "openai/whisper-small")),
            freeze=bool(acfg.get("freeze", True)),
        )
        d_audio = self.audio_encoder.d_audio

        # ── Видео-энкодер ────────────────────────────────────────────────────
        vcfg = mcfg.video_encoder
        self.video_encoder = VideoEncoder(
            d_model=int(vcfg.get("d_video", d_model)),
            n_layers=int(vcfg.get("n_layers", 4)),
            n_heads=int(vcfg.get("n_heads", 8)),
        )
        d_video = self.video_encoder.d_model

        # ── Fusion ───────────────────────────────────────────────────────────
        fcfg = mcfg.fusion
        fusion_type = str(fcfg.get("type", "cross_attention"))

        if mode == "audio_only":
            # Нет fusion — просто линейная проекция аудио в d_model
            self.fusion = None
            self.audio_proj = (
                nn.Linear(d_audio, d_model) if d_audio != d_model else nn.Identity()
            )
        elif mode == "video_only":
            # Нет fusion — просто проекция видео в d_model (если нужна)
            self.fusion = None
            self.video_proj = (
                nn.Linear(d_video, d_model) if d_video != d_model else nn.Identity()
            )
        else:  # "av"
            if fusion_type == "concat":
                self.fusion = ConcatFusion(
                    d_audio=d_audio,
                    d_video=d_video,
                    d_model=d_model,
                )
            else:  # cross_attention
                self.fusion = CrossAttentionFusion(
                    d_audio=d_audio,
                    d_video=d_video,
                    d_model=d_model,
                    n_heads=int(fcfg.get("n_heads", 8)),
                    n_layers=int(fcfg.get("n_layers", 2)),
                    dropout=float(fcfg.get("dropout", 0.1)),
                )

        # ── CTC-голова ───────────────────────────────────────────────────────
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # Инициализация CTC-головы: маленькие веса + bias=0
        nn.init.normal_(self.ctc_head.weight, std=0.02)
        nn.init.zeros_(self.ctc_head.bias)

    def forward(
        self,
        audio_mel: torch.Tensor,
        audio_lens: torch.Tensor,
        video: torch.Tensor,
        video_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_mel:  (B, 80, T_mel)         — лог-мел спектрограммы
            audio_lens: (B,)                   — реальные длины в мел-фреймах
            video:      (B, T_v, 1, 96, 96)    — кропы губ
            video_lens: (B,)                   — реальные длины в кадрах

        Returns:
            logits:   (B, T_out, vocab_size)   — необработанные логиты для CTC
            out_lens: (B,)                     — реальные длины по временной оси
        """
        # ── Прогон через энкодеры ─────────────────────────────────────────
        a_feats, a_lens = self.audio_encoder(audio_mel, audio_lens)
        # a_feats: (B, T_a, d_audio)

        v_feats, v_lens = self.video_encoder(video, video_lens)
        # v_feats: (B, T_v, d_video)

        # ── Modality dropout (только при обучении) ────────────────────────
        if self.training and self.modality_dropout > 0.0:
            r = torch.rand(1).item()
            if r < self.modality_dropout:
                # Обнуляем аудио → модель учится работать только на видео
                a_feats = torch.zeros_like(a_feats)
            elif r < 2.0 * self.modality_dropout:
                # Обнуляем видео → модель учится работать только на аудио
                v_feats = torch.zeros_like(v_feats)

        # ── Fusion ────────────────────────────────────────────────────────
        if self.mode == "audio_only":
            fused = self.audio_proj(a_feats)
            out_lens = a_lens
        elif self.mode == "video_only":
            fused = self.video_proj(v_feats)
            out_lens = v_lens
        else:
            fused, out_lens = self.fusion(a_feats, a_lens, v_feats, v_lens)
        # fused: (B, T_out, d_model)

        # ── CTC-голова ────────────────────────────────────────────────────
        logits = self.ctc_head(fused)  # (B, T_out, vocab_size)

        return logits, out_lens

    @torch.no_grad()
    def transcribe(
        self,
        audio_mel: torch.Tensor,
        audio_lens: torch.Tensor,
        video: torch.Tensor,
        video_lens: torch.Tensor,
        tokenizer,
    ) -> list[str]:
        """
        Удобный метод для инференса: возвращает декодированный текст.

        Args:
            tokenizer: экземпляр CharTokenizer с методом decode().
        """
        self.eval()
        logits, out_lens = self.forward(audio_mel, audio_lens, video, video_lens)
        # Жадное CTC-декодирование
        pred_ids = logits.argmax(dim=-1)  # (B, T_out)
        results = []
        for i, length in enumerate(out_lens.tolist()):
            ids = pred_ids[i, :length].tolist()
            text = tokenizer.decode(ids, collapse_blanks=True)
            results.append(text)
        return results


def build_model(cfg, vocab_size: int) -> AVSRModel:
    """
    Фабричная функция: строит AVSRModel из OmegaConf конфига.

    Использование:
        from src.utils.config import load_config
        from src.models.avsr_model import build_model
        from src.data.dataset import CharTokenizer

        cfg = load_config("configs/avsr_baseline.yaml")
        tokenizer = CharTokenizer()
        model = build_model(cfg, tokenizer.vocab_size)
    """
    return AVSRModel(cfg, vocab_size=vocab_size)
