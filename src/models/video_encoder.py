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

# TODO(day-9): импорты — torch, torch.nn, torchvision.models.resnet18

# TODO(day-9): класс VideoFrontend(nn.Module) — 3D-conv + ResNet-18
#   __init__:
#     — Conv3d(in=1, out=64, kernel=(5,7,7), stride=(1,2,2), padding=(2,3,3))
#     — BatchNorm3d(64), ReLU
#     — MaxPool3d(kernel=(1,3,3), stride=(1,2,2), padding=(0,1,1))
#     — resnet18(weights=None), убрать первый conv1/maxpool/avgpool/fc
#     — оставить только layer1..layer4, на выходе adaptive_avg_pool2d(1)
#   forward(x: (B, T, 1, 96, 96)) -> (B, T, 512)

# TODO(day-9): класс VideoEncoder(nn.Module)
#   __init__(d_model=512, n_layers=4, n_heads=8):
#     — self.frontend = VideoFrontend()
#     — self.proj = nn.Linear(512, d_model)
#     — self.pos = SinusoidalPosEmbedding(d_model)  (или nn.Embedding)
#     — self.transformer = nn.TransformerEncoder(
#         nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True),
#         num_layers=n_layers)
#   forward(video: (B, T, 1, 96, 96), lens: (B,)) -> (B, T, d_model)
#     — feats = self.frontend(video)
#     — feats = self.proj(feats) + pos
#     — mask = (arange(T) >= lens[:,None])  # True там, где паддинг
#     — out = self.transformer(feats, src_key_padding_mask=mask)
