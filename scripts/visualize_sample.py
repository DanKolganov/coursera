"""
Визуализация одного примера: кропы губ + мел-спектрограмма.

Запуск:
    python scripts/visualize_sample.py \\
        --video path/to/sample.mp4 \\
        --audio path/to/sample.wav \\
        --out   visualization.png

Если --audio не указан, попробуем извлечь аудио прямо из видео через
torchaudio.io.StreamReader.

Что увидим на выходе (PNG):
  - Сверху: 8 равномерно выбранных кадров губ (как они приходят в модель).
  - Снизу: лог-мел спектрограмма (тепловая карта).

Это контрольная точка дня 3 из плана: убедиться, что предобработка
работает корректно, прежде чем строить датасет и модель.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


# Чтобы импорт `from src...` работал при запуске из корня проекта
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import (   # noqa: E402
    LipROIExtractor,
    SAMPLE_RATE,
    video_to_lip_tensor,
    waveform_to_mel,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("visualize")


def load_audio(audio_path: Path | None, video_path: Path) -> tuple[torch.Tensor, int]:
    """Загружает аудио. Если audio_path не задан — берём звуковую дорожку из видео."""
    if audio_path is not None:
        wav, sr = torchaudio.load(str(audio_path))
        log.info("Загрузил аудио: %s (sr=%d, длина=%.2f сек)",
                 audio_path.name, sr, wav.shape[-1] / sr)
        return wav, sr
    # fallback: вытаскиваем звук из mp4
    log.info("Аудио-файл не задан, извлекаем дорожку из %s", video_path.name)
    wav, sr = torchaudio.load(str(video_path))
    log.info("Звуковая дорожка: sr=%d, длина=%.2f сек", sr, wav.shape[-1] / sr)
    return wav, sr


def plot_results(
    lips: torch.Tensor,        # (T, 1, 96, 96), float32 in [0, 1]
    mel: torch.Tensor,         # (80, T_mel) log-mel
    out_path: Path,
    n_lip_frames: int = 8,
) -> None:
    """Рисует кадры губ и спектрограмму, сохраняет PNG."""
    import matplotlib
    matplotlib.use("Agg")  # без GUI — нужно для серверов без X11
    import matplotlib.pyplot as plt

    T = lips.shape[0]
    n = min(n_lip_frames, T)
    # Равномерно выбираем n кадров из T
    idx = np.linspace(0, T - 1, n).astype(int)
    frames = lips[idx, 0].numpy()         # (n, 96, 96)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [1, 1.5]},
    )

    # Верхняя строка: лента из n кадров губ
    strip = np.concatenate([frames[i] for i in range(n)], axis=1)
    axes[0].imshow(strip, cmap="gray", aspect="auto")
    axes[0].set_title(
        f"Кропы губ (выбрано {n} кадров из {T}). "
        f"Каждый кадр 96×96, серый.",
        fontsize=11,
    )
    axes[0].axis("off")

    # Нижняя строка: спектрограмма
    mel_np = mel.numpy()
    im = axes[1].imshow(
        mel_np, aspect="auto", origin="lower",
        interpolation="nearest", cmap="magma",
    )
    axes[1].set_title(
        f"Лог-мел спектрограмма: {mel_np.shape[0]} полос × {mel_np.shape[1]} фреймов "
        f"(шаг 10 мс ⇒ длительность {mel_np.shape[1] / 100:.2f} сек)",
        fontsize=11,
    )
    axes[1].set_xlabel("Время (фреймы по 10 мс)")
    axes[1].set_ylabel("Мел-полосы (низкие частоты внизу)")
    plt.colorbar(im, ax=axes[1], label="log(power)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    log.info("Сохранил визуализацию в %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True,
                        help="путь к видеофайлу (mp4/mov/avi)")
    parser.add_argument("--audio", type=Path, default=None,
                        help="необязательно: отдельный wav. Если не задан — "
                             "вытащим звук из видео")
    parser.add_argument("--out", type=Path, default=Path("visualization.png"),
                        help="куда сохранить PNG")
    args = parser.parse_args()

    if not args.video.exists():
        log.error("Видео не найдено: %s", args.video)
        return 1

    # --- 1. Видео -> губы ---
    log.info("Извлекаю кропы губ из %s ...", args.video)
    with LipROIExtractor() as ext:
        lip_tensor, stats = video_to_lip_tensor(
            args.video, extractor=ext, return_stats=True
        )
    log.info(
        "Видео: %d кадров всего, %d с лицом (%.1f%%), FPS=%.1f",
        stats["total_frames"], stats["frames_with_face"],
        100.0 * stats["frames_with_face"] / max(stats["total_frames"], 1),
        stats["fps"],
    )
    log.info("Тензор губ: %s, dtype=%s", tuple(lip_tensor.shape), lip_tensor.dtype)

    # --- 2. Аудио -> лог-мел ---
    wav, sr = load_audio(args.audio, args.video)
    mel = waveform_to_mel(wav, sample_rate=sr)
    log.info("Лог-мел: %s, dtype=%s, диапазон [%.2f, %.2f]",
             tuple(mel.shape), mel.dtype, float(mel.min()), float(mel.max()))

    # --- 3. Sanity-check выравнивания ---
    audio_dur = mel.shape[1] / 100.0          # 10 мс на фрейм
    video_dur = stats["frames_with_face"] / max(stats["fps"], 1.0)
    log.info("Длительность по аудио: %.2f сек, по видео: %.2f сек "
             "(расхождение нормально, мы не паддим под одинаковую длину здесь)",
             audio_dur, video_dur)

    # --- 4. Визуализация ---
    plot_results(lip_tensor, mel, args.out)
    log.info("Готово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
