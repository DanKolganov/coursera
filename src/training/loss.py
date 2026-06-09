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

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWrapper(nn.Module):
    """
    Тонкая обёртка над torch CTCLoss.

    Скрывает бойлерплейт:
      - transpose (B, T, V) → (T, B, V)
      - log_softmax по последней оси
      - zero_infinity=True по умолчанию (защита от NaN на коротких примерах)

    Args:
        blank_id:       индекс blank-токена в алфавите (обычно 0 для CTC).
        zero_infinity:  если True — бесконечные loss'ы заменяются нулём
                        (torch-рекомендация для стабильности).
        reduction:      "mean" (по умолчанию) или "sum".
    """

    def __init__(
        self,
        blank_id: int = 0,
        zero_infinity: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.blank_id = blank_id
        self.ctc = nn.CTCLoss(
            blank=blank_id,
            zero_infinity=zero_infinity,
            reduction=reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        out_lens: torch.Tensor,
        text_ids_flat: torch.Tensor,
        text_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Вычисляет CTC loss.

        Args:
            logits:        (B, T, V) — сырые логиты из CTC-головы.
            out_lens:      (B,)      — длины по T (реальные, без паддинга).
            text_ids_flat: (sum_L,)  — конкатенированные целевые последовательности.
            text_lens:     (B,)      — длины каждой целевой последовательности.

        Returns:
            Скалярный loss.

        Замечание:
            CTCLoss требует, чтобы T >= max(text_lens), иначе бросает ошибку
            (или возвращает inf при zero_infinity=True). Если это случается —
            нужно увеличить длину входа или уменьшить batch_size.
        """
        # (B, T, V) → (T, B, V): такой формат ждёт torch CTCLoss
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)

        # Все тензоры должны быть на CPU для CTCLoss (PyTorch особенность)
        # out_lens и text_lens нужны как int32 или int64 на CPU
        loss = self.ctc(
            log_probs,
            text_ids_flat.to(torch.int32),
            out_lens.to(torch.int32),
            text_lens.to(torch.int32),
        )
        return loss


class SpecAugment(nn.Module):
    """
    SpecAugment: маскирование полос частот и времени в мел-спектрограмме.

    Применяется ТОЛЬКО во время обучения. В инференсе — идентичное преобразование.

    Ref: Park et al. "SpecAugment: A Simple Data Augmentation Method
         for Automatic Speech Recognition" (2019).

    Args:
        freq_mask_param: максимальный размер маски по частоте (F).
        time_mask_param: максимальный размер маски по времени (T).
        num_freq_masks:  количество частотных масок.
        num_time_masks:  количество временных масок.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 10,
        num_freq_masks: int = 1,
        num_time_masks: int = 2,
    ) -> None:
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T) — лог-мел спектрограммы.

        Returns:
            Тензор той же формы с замаскированными полосами.
        """
        if not self.training:
            return mel

        mel = mel.clone()
        B, n_mels, T = mel.shape

        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            if f == 0:
                continue
            f0 = torch.randint(0, max(1, n_mels - f), (1,)).item()
            mel[:, f0: f0 + f, :] = 0.0

        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            if t == 0:
                continue
            t0 = torch.randint(0, max(1, T - t), (1,)).item()
            mel[:, :, t0: t0 + t] = 0.0

        return mel
