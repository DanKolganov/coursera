"""
Аудио-энкодер: обёртка над предобученным Whisper.

Стратегия: берём whisper-small из transformers, выкидываем декодер
(он нам не нужен — у нас своя CTC-голова), оставляем только энкодер.
Замораживаем веса, чтобы:
  1) экономить память (нельзя обучать всё на бесплатной GPU);
  2) гарантировать, что аудио-ветка работает на уровне Whisper из коробки;
  3) ускорить обучение в разы.

Whisper-encoder ожидает мел-спектрограмму (B, 80, 3000) — ровно 30 сек.
У нас же длина переменная (≤15 сек). Решение: паддить нулями до 3000
и потом обрезать выход по реальной длине.
"""
from __future__ import annotations

# TODO(day-8): импорты — torch, transformers.WhisperModel

# TODO(day-8): класс AudioEncoder(nn.Module)
#   __init__(self, model_name="openai/whisper-small", freeze=True):
#     — загрузить WhisperModel.from_pretrained(model_name).encoder
#     — если freeze: for p in self.parameters(): p.requires_grad = False
#     — d_audio = encoder.config.d_model  (для whisper-small это 768)
#
#   forward(self, mel: Tensor(B, 80, T)) -> Tensor(B, T', d_audio):
#     — паддинг/обрезка по последней оси до 3000
#     — encoder(mel).last_hidden_state -> (B, 1500, 768)
#     — обрезать по реальной длине (T_audio_real)
#     — вернуть выход и реальные длины после downsampling
