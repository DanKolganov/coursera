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

from typing import Tuple

import torch
import torch.nn as nn
from transformers import WhisperModel


class AudioEncoder(nn.Module):
    """
    Обёртка над Whisper encoder.

    Args:
        model_name: HuggingFace model id, например "openai/whisper-small".
        freeze:     если True — замораживаем все параметры энкодера.

    Входной тензор: (B, 80, T_mel) — переменная длина T_mel.
    Whisper encoder внутри паддит до 3000, после чего выдаёт (B, 1500, d).
    Мы обрезаем результат до ceil(T_mel / 2) — реального числа фреймов
    после 2-кратного субдискретизирования в свёрточном stem Whisper.

    Выход: (B, T_out, d_audio), out_lens: (B,)
    """

    # Whisper encoder всегда ожидает ровно 3000 мел-фреймов (= 30 сек × 100 Гц)
    WHISPER_MEL_LEN: int = 3000
    # Внутри Whisper два Conv1d(stride=2) → суммарный даунсэмплинг ×2
    WHISPER_STRIDE: int = 2

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        # Загружаем модель целиком и берём только encoder
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder
        del whisper  # декодер нам не нужен

        self.d_audio: int = self.encoder.config.d_model  # 512 для tiny, 768 для small

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self.d_audio

    def forward(
        self,
        mel: torch.Tensor,
        mel_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel:      (B, 80, T_mel)  — лог-мел спектрограммы, паддинг нулями.
            mel_lens: (B,)            — реальные длины T_mel для каждого примера.

        Returns:
            out:      (B, T_out, d_audio)
            out_lens: (B,)  — реальные длины в пространстве выходных фреймов.
        """
        B, n_mels, T = mel.shape

        # --- Паддинг или обрезка до WHISPER_MEL_LEN ---
        if T < self.WHISPER_MEL_LEN:
            pad = self.WHISPER_MEL_LEN - T
            mel = torch.nn.functional.pad(mel, (0, pad), value=0.0)
        else:
            mel = mel[:, :, : self.WHISPER_MEL_LEN]

        # --- Прогон через Whisper encoder ---
        # encoder ожидает (B, n_mels, T) — у нас именно такой формат
        encoder_out = self.encoder(
            input_features=mel,
            return_dict=True,
        )
        # last_hidden_state: (B, 1500, d_audio)
        hidden = encoder_out.last_hidden_state  # (B, 1500, d_audio)

        # --- Обрезаем по реальной длине ---
        # Whisper stride = 2: T_mel фреймов → ceil(T_mel / 2) выходных фреймов
        out_lens = torch.ceil(
            mel_lens.float().clamp(max=self.WHISPER_MEL_LEN) / self.WHISPER_STRIDE
        ).long()
        out_lens = out_lens.clamp(max=hidden.shape[1])

        # hidden уже содержит паддинговые позиции — просто не трогаем их здесь,
        # маскирование делает вышестоящий модуль (Fusion / CTC loss через out_lens)
        return hidden, out_lens
