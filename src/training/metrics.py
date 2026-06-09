"""
Метрики качества + жадное CTC-декодирование.

Содержит:
  - ctc_greedy_decode: argmax + CTC collapse → List[List[int]]
  - decode_batch:      logits → List[str] (использует CharTokenizer)
  - wer:               Word Error Rate через jiwer
  - cer:               Character Error Rate через jiwer
"""
from __future__ import annotations

from typing import List

import torch
import jiwer


def ctc_greedy_decode(
    logits: torch.Tensor,
    out_lens: torch.Tensor,
    blank_id: int = 0,
) -> List[List[int]]:
    """
    Жадное CTC-декодирование: argmax → схлопывание повторов → удаление blank.

    Args:
        logits:   (B, T, V) — сырые логиты (log-probs не нужны, т.к. argmax).
        out_lens: (B,)      — реальные длины по T для каждого примера.
        blank_id: индекс blank-токена (по умолчанию 0).

    Returns:
        Список списков id для каждого примера в батче.
        Пример: [  [7, 5, 12, 12, 15], [8, 5, 12, 12, 15]  ]
    """
    # argmax: (B, T)
    pred_ids = logits.argmax(dim=-1)

    results: List[List[int]] = []
    for b in range(logits.shape[0]):
        length = int(out_lens[b].item())
        ids = pred_ids[b, :length].tolist()

        # CTC collapse: удаляем повторы и blank
        collapsed: List[int] = []
        prev = None
        for tok in ids:
            if tok == blank_id:
                prev = None
                continue
            if tok != prev:
                collapsed.append(tok)
            prev = tok

        results.append(collapsed)
    return results


def decode_batch(
    logits: torch.Tensor,
    out_lens: torch.Tensor,
    tokenizer,
    blank_id: int = 0,
) -> List[str]:
    """
    Жадное декодирование с преобразованием id → текст.

    Args:
        logits:    (B, T, V)
        out_lens:  (B,)
        tokenizer: экземпляр CharTokenizer (с методом decode).
        blank_id:  индекс blank.

    Returns:
        Список строк (по одной на пример в батче).
    """
    ids_list = ctc_greedy_decode(logits, out_lens, blank_id=blank_id)
    return [tokenizer.decode(ids) for ids in ids_list]


def wer(predictions: List[str], references: List[str]) -> float:
    """
    Word Error Rate (WER) через библиотеку jiwer.

    Более низкий WER = лучше. WER=1.0 → 100% слов заменено/пропущено/вставлено.

    Args:
        predictions: список предсказанных транскрипций.
        references:  список эталонных транскрипций.

    Returns:
        Среднее WER по всем парам (float в [0, ∞), обычно < 1.0).
    """
    if not predictions or not references:
        return 0.0
    # jiwer ожидает: wer(reference, hypothesis)
    return float(jiwer.wer(references, predictions))


def cer(predictions: List[str], references: List[str]) -> float:
    """
    Character Error Rate (CER) через библиотеку jiwer.

    Полезна для диагностики: CER < WER, показывает ошибки на уровне символов.

    Args:
        predictions: список предсказанных транскрипций.
        references:  список эталонных транскрипций.

    Returns:
        Среднее CER по всем парам.
    """
    if not predictions or not references:
        return 0.0
    return float(jiwer.cer(references, predictions))


def compute_metrics(
    logits: torch.Tensor,
    out_lens: torch.Tensor,
    reference_texts: List[str],
    tokenizer,
    blank_id: int = 0,
) -> dict:
    """
    Удобная обёртка: за один вызов считает WER и CER.

    Args:
        logits:          (B, T, V)
        out_lens:        (B,)
        reference_texts: список эталонных транскрипций (уже нормализованных).
        tokenizer:       CharTokenizer.

    Returns:
        {"wer": float, "cer": float, "predictions": List[str]}
    """
    preds = decode_batch(logits, out_lens, tokenizer, blank_id=blank_id)
    # Нормализуем референсы так же, как tokenizer нормализует при обучении
    refs = [tokenizer.normalize(t) for t in reference_texts]
    return {
        "wer": wer(preds, refs),
        "cer": cer(preds, refs),
        "predictions": preds,
    }
