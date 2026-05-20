"""
Collator для DataLoader: собирает несколько примеров в батч.

Проблема: у каждого примера своя длина (T_audio, T_video, L_text). PyTorch
требует одинаковый размер внутри батча → паддим до максимума и запоминаем
реальные длины (нужны для CTC и для маскирования attention).
"""
from __future__ import annotations
from typing import List, Dict

# TODO(day-4): импорты — torch

# TODO(day-4): функция avsr_collate_fn(batch: List[dict]) -> dict
#   batch — список словарей из AVSRDataset.__getitem__
#   возвращает:
#     {"audio_mel":   Tensor(B, 80, T_a_max),
#      "audio_lens":  Tensor(B,),
#      "video":       Tensor(B, T_v_max, 1, 96, 96),
#      "video_lens":  Tensor(B,),
#      "text_ids":    Tensor(sum(L_i),)   # flatten для CTC
#      "text_lens":   Tensor(B,)
#      "ids":         List[str]}
#   паддинг: torch.nn.functional.pad или torch.nn.utils.rnn.pad_sequence
