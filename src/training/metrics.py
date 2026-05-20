"""
Метрики качества + жадное CTC-декодирование.
"""
from __future__ import annotations
from typing import List

# TODO(day-11): импорты — torch, jiwer

# TODO(day-11): функция ctc_greedy_decode(logits: (B,T,V), out_lens) -> List[List[int]]
#   argmax по последней оси, схлопывание повторов, удаление blank (id=0)

# TODO(day-11): wer(preds: List[str], refs: List[str]) -> float
#   return jiwer.wer(refs, preds)

# TODO(day-11): cer(preds, refs) -> float
#   return jiwer.cer(refs, preds)
