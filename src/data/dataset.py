"""
PyTorch Dataset для AVSR.

Идея: один пример = одно высказывание. У нас есть манифест-файл (JSON), где
для каждого примера хранятся пути к аудио, видео и текст транскрипции.

Формат манифеста (один пример = одна строка JSON):
  {"id": "abc123",
   "audio": "data/raw/.../abc123.wav",
   "video": "data/raw/.../abc123.mp4",
   "lip_npy": "data/processed/abc123_lips.npy",  # предвычисленный кроп губ
   "text": "hello world",
   "duration": 3.21}

Длина: ограничиваем 15 секундами, остальное выкидываем (память + скорость).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

# TODO(day-4): импорты — torch, json, numpy, soundfile

# TODO(day-4): класс AVSRDataset(torch.utils.data.Dataset)
#   __init__(self, manifest_path, max_duration=15.0,
#            tokenizer=None, augment=False)
#     — загрузить манифест построчно (jsonl)
#     — отфильтровать по max_duration
#     — сохранить tokenizer (нужен в __getitem__ для конвертации текста в id)
#
#   __len__ -> int
#
#   __getitem__(idx) -> Dict[str, Tensor]
#     — загрузить аудио: soundfile.read → torch.tensor → waveform_to_mel
#     — загрузить губы: np.load(lip_npy) → torch.tensor(T, 1, 96, 96) / 255
#     — токенизировать текст: List[int] символов алфавита
#     — (если augment) — наложить шум, SpecAugment, отразить видео
#     — вернуть dict:
#         {"audio_mel": Tensor(80, T_a),
#          "video": Tensor(T_v, 1, 96, 96),
#          "text_ids": Tensor(L,),
#          "text": str,           # для отладки
#          "id": str}

# TODO(day-4): простой токенизатор на уровне символов
#   class CharTokenizer:
#     алфавит = "<blank>" + "abcdefghijklmnopqrstuvwxyz '"
#     encode(text) -> List[int]
#     decode(ids) -> str
