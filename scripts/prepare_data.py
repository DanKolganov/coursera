"""
Препроцессинг датасета: губы → .npy, сборка манифеста.

Запуск:
    # LRS3 или любой датасет с парами audio/video
    python scripts/prepare_data.py \
        --src data/raw/lrs3/trainval \
        --dst data/processed/lips \
        --manifest-out data/manifests/train.jsonl \
        --split train \
        --audio-ext .wav \
        --video-ext .mp4 \
        --text-ext .txt

    # Параллельный запуск на N ядрах CPU
    python scripts/prepare_data.py ... --num-workers 8

Ожидаемая структура src-директории (один из вариантов):
    trainval/
        AAAA/
            00001.wav (или .mp4 с аудио)
            00001.mp4
            00001.txt   ← транскрипция (одна строка)
        BBBB/
            ...

Или плоская структура:
    trainval/
        utt001.wav
        utt001.mp4
        utt001.txt

Скрипт рекурсивно ищет все .mp4 файлы и подбирает к ним .wav и .txt
с тем же базовым именем (stem).
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Добавляем корень проекта в sys.path, чтобы импорты src.* работали
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import LipROIExtractor, video_to_lip_tensor
from src.utils.logging import get_logger


log = get_logger("avsr.prepare_data")


# =============================================================================
# Один пример: видео → .npy кроп губ + запись для манифеста
# =============================================================================

def process_one(
    video_path: Path,
    dst_dir: Path,
    audio_ext: str = ".wav",
    text_ext: str = ".txt",
) -> Optional[dict]:
    """
    Обрабатывает одно видео: извлекает губы, со��раняет .npy, возвращает запись манифеста.

    Args:
        video_path: путь к видеофайлу.
        dst_dir:    куда сохранять .npy (сохраняем с тем же относительным путём).
        audio_ext:  расширение аудио-файла.
        text_ext:   расширение файла транскрипции.

    Returns:
        Словарь-запись для манифеста или None при ошибке.
    """
    stem = video_path.stem
    parent = video_path.parent

    # --- Аудио-файл ---
    audio_path = parent / (stem + audio_ext)
    if not audio_path.exists():
        # Попробуем .wav если audio_ext — .mp4 (аудио внутри видео)
        # Fallback: ffmpeg-extracted wav должен быть рядом
        log.warning("Аудио не найдено: %s", audio_path)
        return None

    # --- Текст ---
    text_path = parent / (stem + text_ext)
    if not text_path.exists():
        log.warning("Транскрипция не найдена: %s", text_path)
        return None

    with open(text_path, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return None

    # --- Длительность аудио ---
    try:
        with sf.SoundFile(str(audio_path)) as sf_file:
            duration = len(sf_file) / sf_file.samplerate
    except Exception as e:
        log.warning("Не могу прочитать аудио %s: %s", audio_path, e)
        return None

    # --- Кроп губ ---
    # Путь для .npy: dst_dir / <уникальный id>.npy
    # Делаем id из относительного пути чтобы не было коллизий
    rel_stem = str(video_path).replace(os.sep, "_").replace("/", "_")
    lip_npy_path = dst_dir / (rel_stem + "_lips.npy")
    lip_npy_path.parent.mkdir(parents=True, exist_ok=True)

    if not lip_npy_path.exists():
        try:
            tensor, stats = video_to_lip_tensor(video_path, return_stats=True)
            # Сохраняем как uint8 (T, 96, 96) для экономии места
            arr = (tensor.squeeze(1).numpy() * 255).astype(np.uint8)
            np.save(str(lip_npy_path), arr)
            log.debug(
                "Сохранено %s: %d кадров, %d пропущено",
                lip_npy_path.name, stats["frames_with_face"], stats["missing_frames"],
            )
        except RuntimeError as e:
            log.warning("Не нашли лицо в %s: %s", video_path.name, e)
            return None
        except Exception as e:
            log.warning("Ошибка при обработке %s: %s", video_path.name, e)
            return None

    return {
        "id": stem,
        "audio": str(audio_path),
        "video": str(video_path),
        "lip_npy": str(lip_npy_path),
        "text": text,
        "duration": round(duration, 3),
    }


def process_one_star(args: tuple) -> Optional[dict]:
    """Обёртка для multiprocessing (unpacking args)."""
    return process_one(*args)


# =============================================================================
# Главная функция
# =============================================================================

def process_from_manifest(
    manifest_in: Path,
    manifest_out: Path,
    dst_dir: Path,
    num_workers: int = 1,
    max_files: Optional[int] = None,
) -> None:
    """
    Режим --manifest-in: берёт готовый JSONL-манифест (с полем video),
    извлекает кропы губ и записывает новый манифест с заполненным lip_npy.

    Используется когда манифест уже собран (например из MUAVIC TSV),
    и нужно только добавить MediaPipe-обработку.
    """
    records_in = []
    with open(manifest_in, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records_in.append(json.loads(line))

    if max_files:
        records_in = records_in[:max_files]

    log.info("manifest-in режим: %d примеров из %s", len(records_in), manifest_in)

    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    def process_record(rec: dict) -> Optional[dict]:
        video_path = Path(rec["video"])
        if not video_path.exists():
            log.warning("Видео не найдено: %s", video_path)
            return None

        # Если lip_npy уже заполнен и файл существует — пропускаем
        existing = rec.get("lip_npy", "")
        if existing and Path(existing).exists():
            return rec

        rel_stem = str(video_path).replace(os.sep, "_").replace("/", "_")
        lip_npy_path = dst_dir / (rel_stem + "_lips.npy")
        lip_npy_path.parent.mkdir(parents=True, exist_ok=True)

        if not lip_npy_path.exists():
            try:
                tensor, stats = video_to_lip_tensor(video_path, return_stats=True)
                arr = (tensor.squeeze(1).numpy() * 255).astype(np.uint8)
                np.save(str(lip_npy_path), arr)
            except Exception as e:
                log.warning("Ошибка %s: %s", video_path.name, e)
                return None

        out = dict(rec)
        out["lip_npy"] = str(lip_npy_path)
        return out

    records_out = []
    errors = 0

    if num_workers <= 1:
        for rec in tqdm(records_in, desc="MediaPipe"):
            result = process_record(rec)
            if result is not None:
                records_out.append(result)
            else:
                errors += 1
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_record, r): r for r in records_in}
            for future in tqdm(as_completed(futures), total=len(futures), desc="MediaPipe"):
                result = future.result()
                if result is not None:
                    records_out.append(result)
                else:
                    errors += 1

    with open(manifest_out, "w", encoding="utf-8") as f:
        for rec in records_out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info("Готово: %d / %d  (%d ошибок)", len(records_out), len(records_in), errors)
    log.info("Манифест записан: %s", manifest_out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Препроцессинг аудио-видео датасета для AVSR"
    )
    # Режим A: директория с сырыми файлами
    parser.add_argument(
        "--src", default=None,
        help="Папка с сырым датасетом (рекурсивный поиск .mp4). "
             "Взаимоисключает с --manifest-in."
    )
    # Режим B: готовый манифест (MUAVIC workflow)
    parser.add_argument(
        "--manifest-in", default=None,
        help="Входной JSONL-манифест с полем video. "
             "Взаимоисключает с --src."
    )
    parser.add_argument(
        "--dst", required=True,
        help="Папка для сохранения .npy кропов губ"
    )
    parser.add_argument(
        "--manifest-out", required=True,
        help="Путь к выходному .jsonl манифесту"
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train",
        help="Тип сплита (только для логирования)"
    )
    parser.add_argument(
        "--audio-ext", default=".wav",
        help="Расширение аудио-файлов (default: .wav)"
    )
    parser.add_argument(
        "--video-ext", default=".mp4",
        help="Расширение видео-файлов (default: .mp4)"
    )
    parser.add_argument(
        "--text-ext", default=".txt",
        help="Расширение файлов транскрипций (default: .txt)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Число параллельных процессов (default: 1, на Colab лучше 1-2)"
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Ограничить кол-во файлов (для отладки)"
    )
    args = parser.parse_args()

    if args.manifest_in and args.src:
        log.error("Нельзя использовать --src и --manifest-in одновременно")
        sys.exit(1)
    if not args.manifest_in and not args.src:
        log.error("Нужно указать --src или --manifest-in")
        sys.exit(1)

    # ── Режим B: manifest-in (MUAVIC workflow) ────────────────────────────────
    if args.manifest_in:
        process_from_manifest(
            manifest_in=Path(args.manifest_in),
            manifest_out=Path(args.manifest_out),
            dst_dir=Path(args.dst),
            num_workers=args.num_workers,
            max_files=args.max_files,
        )
        return

    # ── Режим A: src директория ───────────────────────────────────────────────
    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    manifest_path = Path(args.manifest_out)

    if not src_dir.exists():
        log.error("Директория не найдена: %s", src_dir)
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Собираем список видео ---
    video_paths = sorted(src_dir.rglob(f"*{args.video_ext}"))
    if not video_paths:
        log.error("Видео-файлы не найдены в %s с расширением %s", src_dir, args.video_ext)
        sys.exit(1)

    if args.max_files:
        video_paths = video_paths[: args.max_files]

    log.info(
        "Сплит '%s': найдено %d видео-файлов, запускаю на %d процессах",
        args.split, len(video_paths), args.num_workers,
    )

    # --- Обрабатываем ---
    task_args = [
        (vp, dst_dir, args.audio_ext, args.text_ext)
        for vp in video_paths
    ]

    records = []
    errors = 0

    if args.num_workers <= 1:
        # Однопоточный режим (удобнее для отладки)
        for task in tqdm(task_args, desc=f"Prepare {args.split}"):
            result = process_one(*task)
            if result is not None:
                records.append(result)
            else:
                errors += 1
    else:
        # Многопроцессорный режим
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_one_star, t): t for t in task_args}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Prepare {args.split}",
            ):
                result = future.result()
                if result is not None:
                    records.append(result)
                else:
                    errors += 1

    # --- Записываем манифест ---
    # Сортируем по id для воспроизводим��сти
    records.sort(key=lambda r: r["id"])

    with open(manifest_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_duration = sum(r["duration"] for r in records)
    log.info(
        "Манифест записан: %s",
        manifest_path,
    )
    log.info(
        "  Успешно: %d / %d  (%.1f%% ошибок)",
        len(records), len(video_paths),
        100.0 * errors / max(len(video_paths), 1),
    )
    log.info(
        "  Суммарная длительность: %.1f ч (%.0f мин)",
        total_duration / 3600, total_duration / 60,
    )

    # Статистика по длительности
    durations = np.array([r["duration"] for r in records])
    if len(durations) > 0:
        log.info(
            "  Длительность: min=%.1fs, max=%.1fs, mean=%.1fs, median=%.1fs",
            durations.min(), durations.max(), durations.mean(), np.median(durations),
        )


if __name__ == "__main__":
    main()
