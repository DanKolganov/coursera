"""
Предобработка аудио и видео — превращаем сырьё в тензоры для модели.

Содержит:
  - waveform_to_mel:   raw audio  -> log-mel spectrogram (80, T)
  - LipROIExtractor:   класс, извлекающий кроп губ 96x96 из одного кадра
  - video_to_lip_tensor: читает видео и возвращает тензор (T, 1, 96, 96)

ВАЖНО: предобработку губ дорого пересчитывать на каждой эпохе
(MediaPipe работает на CPU). Поэтому в проекте мы один раз прогоняем
весь датасет скриптом scripts/prepare_data.py и кешируем .npy-файлы
на диск.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torchaudio


# =============================================================================
# Константы (вынесены наверх, чтобы при необходимости менять в одном месте)
# =============================================================================

# --- Аудио ---
SAMPLE_RATE: int = 16_000       # частота дискретизации речи
N_FFT: int = 512                # размер FFT (степень двойки для скорости)
WIN_LENGTH: int = 400           # окно STFT: 25 мс при 16 кГц
HOP_LENGTH: int = 160           # шаг STFT: 10 мс -> 100 фреймов в секунду
N_MELS: int = 80                # число мел-полос

# --- Видео ---
LIP_SIZE: int = 96              # сторона квадратного кропа губ
LIP_PADDING: float = 0.2        # доп. поля вокруг bbox губ (20%)

# Индексы ландмарок Face Mesh, относящихся к губам.
# Берём ВНЕШНИЙ и ВНУТРЕННИЙ контуры губ — это даст полный bbox.
# Эти индексы фиксированы в MediaPipe и не меняются от версии к версии.
LIP_LANDMARK_INDICES: list[int] = sorted({
    # Внешний контур
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    # Внутренний контур
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
})


log = logging.getLogger(__name__)


# =============================================================================
# АУДИО
# =============================================================================

def waveform_to_mel(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Преобразует raw waveform в лог-мел спектрограмму.

    Args:
        waveform:    тензор (T,) или (channels, T). Если несколько каналов —
                     усредняем в моно.
        sample_rate: фактическая частота waveform. Если не 16 кГц — ресэмплим.
        normalize:   если True, делим по пиковой амплитуде (защита от
                     слишком тихих/громких записей).

    Returns:
        Тензор (N_MELS=80, T_mel), где T_mel ≈ T / HOP_LENGTH.
        Значения — log(power + eps), типичный диапазон ~[-20, 5].
    """
    # 1) сводим к моно
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.ndim != 1:
        raise ValueError(f"Ожидался 1D или 2D тензор, получил ndim={waveform.ndim}")

    # 2) приводим к float32 (некоторые wav читаются как int16)
    if waveform.dtype != torch.float32:
        waveform = waveform.float()
        # int16 диапазон [-32768, 32767] -> [-1, 1]
        if waveform.abs().max() > 1.5:
            waveform = waveform / 32768.0

    # 3) ресэмплинг
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE
        )

    # 4) нормализация амплитуды (опционально)
    if normalize:
        peak = waveform.abs().max()
        if peak > 1e-6:
            waveform = waveform / peak

    # 5) лог-мел через torchaudio
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,   # spectrogram of power, not amplitude
        center=True,
    )
    mel = mel_transform(waveform)            # (N_MELS, T_mel)
    log_mel = torch.log(mel + 1e-9)
    return log_mel


# =============================================================================
# ВИДЕО
# =============================================================================

class LipROIExtractor:
    """
    Извлекает кроп губ из BGR-кадра через MediaPipe Face Mesh.

    Использование:
        ext = LipROIExtractor()
        for frame_bgr in video_frames:
            lip = ext.extract(frame_bgr)   # np.ndarray (96, 96) uint8 или None
        ext.close()

    Лучше использовать как context manager:
        with LipROIExtractor() as ext:
            ...

    Внимание: MediaPipe Face Mesh держит внутри C++ ресурсы — не забывай
    вызывать .close() (или используй with).
    """

    def __init__(
        self,
        lip_size: int = LIP_SIZE,
        padding: float = LIP_PADDING,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.lip_size = lip_size
        self.padding = padding
        # static_image_mode=False — лучше для видео (использует tracking
        # между кадрами, быстрее и стабильнее)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Извлекает квадратный кроп губ из одного кадра.

        Args:
            frame_bgr: np.ndarray (H, W, 3), uint8, BGR (как из cv2.imread/cap)

        Returns:
            np.ndarray (lip_size, lip_size) uint8 — серый кроп губ,
            или None, если лицо не найдено.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]

        # MediaPipe ждёт RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # Координаты губных ландмарок в пикселях
        lip_pts = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in LIP_LANDMARK_INDICES],
            dtype=np.float32,
        )

        # Bounding box
        x_min, y_min = lip_pts.min(axis=0)
        x_max, y_max = lip_pts.max(axis=0)

        # Делаем квадратным и добавляем 20% поля
        bw = x_max - x_min
        bh = y_max - y_min
        side = max(bw, bh) * (1.0 + 2.0 * self.padding)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0

        x0 = int(np.clip(cx - side / 2.0, 0, w - 1))
        y0 = int(np.clip(cy - side / 2.0, 0, h - 1))
        x1 = int(np.clip(cx + side / 2.0, 1, w))
        y1 = int(np.clip(cy + side / 2.0, 1, h))

        if x1 - x0 < 8 or y1 - y0 < 8:
            # bbox получился вырожденный (лицо у самого края кадра)
            return None

        crop = frame_bgr[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray, (self.lip_size, self.lip_size), interpolation=cv2.INTER_CUBIC
        )
        return resized

    def close(self) -> None:
        self.face_mesh.close()

    def __enter__(self) -> "LipROIExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def video_to_lip_tensor(
    video_path: str | Path,
    extractor: Optional[LipROIExtractor] = None,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, dict] | torch.Tensor:
    """
    Открывает видеофайл и возвращает тензор губных кропов всех кадров.

    Args:
        video_path:   путь к видео (mp4/mov/avi — что умеет cv2).
        extractor:    если передан, используем его (полезно при батч-обработке,
                      чтобы не пересоздавать MediaPipe на каждый файл).
        return_stats: если True, дополнительно вернёт словарь со статистикой
                      (всего кадров, пропущенных кадров, FPS).

    Returns:
        Тензор (T_видео, 1, 96, 96) float32 в [0, 1].
        Если return_stats=True — кортеж (tensor, stats).
        Кадры, на которых лицо не обнаружено, пропускаются (с логированием).

    Raises:
        FileNotFoundError если видео не открылось.
        RuntimeError если ни на одном кадре лица не нашлось.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV не смог открыть {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_in_header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    own_extractor = extractor is None
    if extractor is None:
        extractor = LipROIExtractor()

    frames: list[np.ndarray] = []
    n_total = 0
    n_missing = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            n_total += 1
            crop = extractor.extract(frame)
            if crop is None:
                n_missing += 1
                continue
            frames.append(crop)
    finally:
        cap.release()
        if own_extractor:
            extractor.close()

    if not frames:
        raise RuntimeError(
            f"В {video_path} не обнаружено ни одного лица "
            f"(проверено {n_total} кадров)."
        )

    if n_missing > 0:
        log.warning(
            "Не нашли лицо в %d из %d кадров (%.1f%%) в %s",
            n_missing, n_total, 100.0 * n_missing / max(n_total, 1), video_path.name,
        )

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0   # (T, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(1)                  # (T, 1, H, W)

    stats = {
        "fps": fps,
        "total_frames": n_total,
        "frames_with_face": len(frames),
        "missing_frames": n_missing,
        "header_total": total_in_header,
    }
    if return_stats:
        return tensor, stats
    return tensor
