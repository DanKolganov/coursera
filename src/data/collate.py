"""
Collate-функция для DataLoader.

Принимает список словарей из AVSRDataset.__getitem__ и собирает в батч,
паддя переменные длины аудио и видео и сохраняя реальные длины для
маскирования и для CTCLoss.

Формат на выходе:
    {
        "audio_mel":  Tensor (B, 80, T_a_max),    float32
        "audio_lens": Tensor (B,),                int64
        "video":      Tensor (B, T_v_max, 1, 96, 96), float32
        "video_lens": Tensor (B,),                int64
        "text_ids":   Tensor (sum L_i,),          int64  ← flatten для CTCLoss
        "text_lens":  Tensor (B,),                int64
        "ids":        List[str],
        "texts":      List[str],
    }
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch


def avsr_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """См. docstring модуля."""
    if not batch:
        raise ValueError("Пустой батч")

    B = len(batch)

    # =====================================================================
    # AUDIO
    # =====================================================================
    # Каждый mel — (n_mels, T_a). Паддим по T_a, n_mels одинаков у всех.
    n_mels = batch[0]["audio_mel"].shape[0]
    audio_lens = torch.tensor(
        [b["audio_mel"].shape[1] for b in batch], dtype=torch.long
    )
    T_a_max = int(audio_lens.max().item())

    audio_mel = torch.zeros(B, n_mels, T_a_max, dtype=torch.float32)
    for i, b in enumerate(batch):
        T = b["audio_mel"].shape[1]
        audio_mel[i, :, :T] = b["audio_mel"]

    # =====================================================================
    # VIDEO
    # =====================================================================
    # Каждое видео — (T_v, C, H, W). Паддим по T_v.
    sample_v = batch[0]["video"]
    C, H, W = sample_v.shape[1], sample_v.shape[2], sample_v.shape[3]
    video_lens = torch.tensor(
        [b["video"].shape[0] for b in batch], dtype=torch.long
    )
    T_v_max = int(video_lens.max().item())

    video = torch.zeros(B, T_v_max, C, H, W, dtype=torch.float32)
    for i, b in enumerate(batch):
        T = b["video"].shape[0]
        video[i, :T] = b["video"]

    # =====================================================================
    # TEXT
    # =====================================================================
    # Для CTC удобно flatten — конкатенировать все таргеты в один длинный
    # тензор + отдельный тензор реальных длин.
    text_lens = torch.tensor(
        [len(b["text_ids"]) for b in batch], dtype=torch.long
    )
    text_ids = torch.cat([b["text_ids"] for b in batch], dim=0) if B > 0 \
        else torch.zeros(0, dtype=torch.long)

    return {
        "audio_mel": audio_mel,
        "audio_lens": audio_lens,
        "video": video,
        "video_lens": video_lens,
        "text_ids": text_ids,
        "text_lens": text_lens,
        "ids": [b["id"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


def make_padding_mask(lens: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """
    Утилита: булева маска паддинга (B, T_max).
    mask[b, t] == True  означает «это паддинг, игнорировать».

    Пример использования с TransformerEncoder:
        mask = make_padding_mask(audio_lens, T_a_max)
        out = transformer(x, src_key_padding_mask=mask)
    """
    if max_len is None:
        max_len = int(lens.max().item())
    arange = torch.arange(max_len, device=lens.device).unsqueeze(0)  # (1, T)
    return arange >= lens.unsqueeze(1)                                # (B, T)
