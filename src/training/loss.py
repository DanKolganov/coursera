"""
CTC loss — то, чем мы будем обучать модель.

Что делает CTC (напомнить себе):
  Модель предсказывает в каждый момент времени распределение по алфавиту
  (буквы + спец-токен blank). CTC автоматически перебирает все возможные
  выравнивания между этой последовательностью предсказаний и эталонным
  текстом, и максимизирует суммарную вероятность.
  Нам как пользователям достаточно знать сигнатуру torch.nn.CTCLoss.

ВАЖНО: torch.nn.CTCLoss ждёт логиты в формате (T, B, C) с log_softmax
поверх C, а не (B, T, C) как у нас. Не забыть transpose и log_softmax.
"""
from __future__ import annotations

# TODO(day-11): импорты — torch, torch.nn, torch.nn.functional

# TODO(day-11): class CTCLossWrapper(nn.Module)
#   __init__(blank_id=0, zero_infinity=True)
#   forward(logits: (B, T, V), out_lens: (B,),
#           text_ids_flat: (sum L,), text_lens: (B,)) -> scalar
#     log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
#     return F.ctc_loss(log_probs, text_ids_flat, out_lens, text_lens,
#                       blank=blank_id, zero_infinity=zero_infinity)
