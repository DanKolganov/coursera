"""
Видео-энкодер: 3D-Conv → ResNet-18 → Transformer.

Идея пошагово:
  1) Входная пачка кадров (B, T, 1, 96, 96) — серые кропы губ.
  2) Первый слой — 3D-свёртка по (T, H, W). Захватывает короткие движения
     (открытие/закрытие рта длится ~3-5 кадров). После неё:
       (B, 64, T, 24, 24)
  3) Применяем ResNet-18 покадрово (свёртки 2D). На выходе на каждый кадр
     получаем вектор 512. Итог: (B, T, 512).
  4) Линейная проекция 512 → d_model (например, 512).
  5) Стек из ~4 слоёв TransformerEncoder для моделирования временных
     зависимостей между кадрами.
  6) Выход: (B, T, d_model).

Веса: можно начать со случайной инициализации, но если время поджимает —
взять предобученные веса из Auto-AVSR (https://github.com/mpc001/auto_avsr).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# =============================================================================
# Sinusoidal positional encoding
# =============================================================================

class SinusoidalPosEncoding(nn.Module):
    """
    Классическое синусоидальное позиционное кодирование (Vaswani et al. 2017).
    Добавляем к эмбеддингу (B, T, d_model).
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # (1, max_len, d_model) — broadcastable по батч-оси
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        return x + self.pe[:, : x.shape[1]]


# =============================================================================
# VideoFrontend: 3D-conv → ResNet-18 backbone
# =============================================================================

class VideoFrontend(nn.Module):
    """
    Извлекает пространственно-временные признаки из последовательности кадров губ.

    Архитектура (следует Auto-AVSR / LipNet conventional frontend):
      1) Conv3d(1→64, kernel=(5,7,7), stride=(1,2,2), padding=(2,3,3))
         → (B, 64, T, 48, 48)
      2) BatchNorm3d + ReLU
      3) MaxPool3d(kernel=(1,3,3), stride=(1,2,2), padding=(0,1,1))
         → (B, 64, T, 24, 24)
      4) ResNet-18 (layer1..layer4) без первых Conv/BN/Pool и без avg_pool/fc
         → на каждый кадр: (B*T, 512, h', w')
      5) AdaptiveAvgPool2d(1) → (B*T, 512)
      6) Reshape → (B, T, 512)

    Вход:  (B, T, 1, 96, 96)
    Выход: (B, T, 512)
    """

    FRONTEND_CHANNELS: int = 64
    RESNET_OUT: int = 512

    def __init__(self) -> None:
        super().__init__()

        # 3D-свёрточный слой: захватывает движение губ на ~5 кадрах
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=self.FRONTEND_CHANNELS,
            kernel_size=(5, 7, 7),
            stride=(1, 2, 2),
            padding=(2, 3, 3),
            bias=False,
        )
        self.bn3d = nn.BatchNorm3d(self.FRONTEND_CHANNELS)
        self.relu = nn.ReLU(inplace=True)
        self.pool3d = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

        # ResNet-18 backbone — берём только layer1..layer4
        backbone = resnet18(weights=None)
        # Адаптируем первый слой: backbone ожидает 3-канальный вход,
        # но мы передаём 64 карты от 3D-conv. Просто заменяем conv1.
        self.layer0 = nn.Sequential(
            nn.Conv2d(
                self.FRONTEND_CHANNELS,
                self.FRONTEND_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.FRONTEND_CHANNELS),
            nn.ReLU(inplace=True),
        )
        self.layer1 = backbone.layer1   # 64 → 64,  stride=1
        self.layer2 = backbone.layer2   # 64 → 128, stride=2
        self.layer3 = backbone.layer3   # 128 → 256, stride=2
        self.layer4 = backbone.layer4   # 256 → 512, stride=2
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 1, H, W) — кропы губ, float32 в [0, 1].

        Returns:
            (B, T, 512)
        """
        B, T, C, H, W = x.shape

        # (B, T, 1, H, W) → (B, 1, T, H, W) — порядок для Conv3d
        x = x.permute(0, 2, 1, 3, 4)  # (B, 1, T, H, W)

        # 3D conv + pool → (B, 64, T, H', W')
        x = self.pool3d(self.relu(self.bn3d(self.conv3d(x))))
        # x.shape: (B, 64, T, ~24, ~24)

        # Переходим к 2D: обрабатываем каждый кадр независимо
        _, C_feat, T_out, H_feat, W_feat = x.shape
        x = x.permute(0, 2, 1, 3, 4)          # (B, T, 64, H', W')
        x = x.contiguous().view(B * T_out, C_feat, H_feat, W_feat)  # (B*T, 64, H', W')

        # ResNet-18 layers
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # (B*T, 512, 1, 1)
        x = x.flatten(1)     # (B*T, 512)

        x = x.view(B, T_out, self.RESNET_OUT)  # (B, T, 512)
        return x


# =============================================================================
# VideoEncoder: frontend + proj + pos + Transformer
# =============================================================================

class VideoEncoder(nn.Module):
    """
    Полный видео-энкодер для AVSR.

    Принимает последовательность кадров губ (B, T, 1, 96, 96),
    возвращает контекстуализированные эмбеддинги (B, T, d_model).

    Args:
        d_model:  размер эмбеддинга (должен совпадать с аудио после fusion).
        n_layers: число слоёв TransformerEncoder.
        n_heads:  число голов attention.
        dropout:  dropout в трансформере.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.frontend = VideoFrontend()

        # Проекция 512 → d_model (если d_model == 512, это просто identity-like)
        self.proj = nn.Linear(VideoFrontend.RESNET_OUT, d_model)
        self.pos_enc = SinusoidalPosEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-norm = более стабильное обучение
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.d_model = d_model

    @property
    def output_dim(self) -> int:
        return self.d_model

    def forward(
        self,
        video: torch.Tensor,
        video_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video:      (B, T, 1, H, W) — кропы губ.
            video_lens: (B,) — реальные длины (без паддинга).

        Returns:
            out:      (B, T, d_model)
            out_lens: (B,)  — те же video_lens (длина не меняется)
        """
        B, T, C, H, W = video.shape

        # 1) Извлекаем признаки через 3D-conv + ResNet
        feats = self.frontend(video)  # (B, T, 512)

        # Примечание: T_out из frontend может чуть отличаться от T
        # из-за MaxPool3d с stride=(1,2,2) — по временной оси stride=1,
        # поэтому T_out == T всегда. Проверяем на всякий случай:
        T_out = feats.shape[1]

        # 2) Проекция + позиционное кодирование
        feats = self.proj(feats)           # (B, T_out, d_model)
        feats = self.dropout(self.pos_enc(feats))

        # 3) Маска паддинга: True = паддинговая позиция (игнорировать)
        arange = torch.arange(T_out, device=feats.device).unsqueeze(0)  # (1, T)
        key_padding_mask = arange >= video_lens.unsqueeze(1)             # (B, T)

        # 4) TransformerEncoder
        out = self.transformer(feats, src_key_padding_mask=key_padding_mask)

        # Длины не изменились
        out_lens = video_lens.clamp(max=T_out)
        return out, out_lens
