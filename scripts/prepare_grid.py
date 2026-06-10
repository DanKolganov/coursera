"""
Подготовка GRID Corpus: скачивание, парсинг, препроцессинг, манифесты.

Использование:
    # Все 33 спикера на бесплатном Drive рискованно — займёт ~50 ГБ.
    # Для смоук-теста — один спикер:
    python scripts/prepare_grid.py --speakers s1 --base data/raw/grid

    # Для рабочей выборки — 10 спикеров:
    python scripts/prepare_grid.py --speakers all10 --base /content/drive/MyDrive/avsr/grid

    # Полный датасет 33 спикера:
    python scripts/prepare_grid.py --speakers all --base /content/drive/MyDrive/avsr/grid

Что делает (всё с возобновлением — повторный запуск пропускает уже
обработанные файлы):
    1. wget из зеркала Sheffield: видео-zip и align-tar для каждого спикера.
    2. unzip / tar -x → s1/video/*.mpg, s1/align/*.align.
    3. ffmpeg → s1/wav/*.wav (16 кГц моно).
    4. MediaPipe FaceLandmarker → s1/lips/*.npy (T, 96, 96 uint8).
    5. Парсинг .align → транскрипция без "sil".
    6. Speaker-disjoint train/val: спикеры val_speakers целиком уходят в val.
    7. data/manifests/train.jsonl, data/manifests/val.jsonl.

Структура внутри base:
    base/
      raw/
        s1/
          video/*.mpg
          align/*.align
        s2/...
      processed/
        s1/
          wav/*.wav
          lips/*.npy
        s2/...
      manifests/
        train.jsonl
        val.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Чтобы import src.* работал
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from src.data.preprocessing import LipROIExtractor, video_to_lip_tensor


log = logging.getLogger("prepare_grid")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")


# =============================================================================
# Список спикеров
# =============================================================================

# Все 33 спикера GRID (s21 отсутствует в оригинальном корпусе)
ALL_SPEAKERS = [
    "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",  "s8",  "s9",  "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20",
    "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",
    "s32", "s33", "s34",
]

PRESETS = {
    "smoke": ["s1"],                                          # один — для проверки
    "all10": ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    "all20": ALL_SPEAKERS[:20],
    "all":   ALL_SPEAKERS,                                    # все 33
}


# =============================================================================
# Скачивание
# =============================================================================

SHEFFIELD_BASE = "https://spandh.dcs.shef.ac.uk/gridcorpus"


def download_speaker(spk: str, raw_dir: Path) -> bool:
    """Скачивает и распаковывает архивы для одного спикера. Идемпотентно."""
    spk_dir = raw_dir / spk
    video_dir = spk_dir / "video"
    align_dir = spk_dir / "align"

    if video_dir.exists() and any(video_dir.iterdir()) and \
       align_dir.exists() and any(align_dir.iterdir()):
        log.info("[%s] уже распакован, пропускаю", spk)
        return True

    spk_dir.mkdir(parents=True, exist_ok=True)

    # Видео
    video_url = f"{SHEFFIELD_BASE}/{spk}/video/{spk}.mpg_vcd.zip"
    video_zip = spk_dir / "video.zip"
    if not video_zip.exists() and not video_dir.exists():
        log.info("[%s] скачиваю видео: %s", spk, video_url)
        r = subprocess.run(["wget", "-q", "-O", str(video_zip), video_url])
        if r.returncode != 0:
            log.error("[%s] не удалось скачать видео", spk)
            video_zip.unlink(missing_ok=True)
            return False
        log.info("[%s] скачано: %.1f МБ", spk,
                 video_zip.stat().st_size / 1024 / 1024)

    # Алайны
    align_url = f"{SHEFFIELD_BASE}/{spk}/align/{spk}.tar"
    align_tar = spk_dir / "align.tar"
    if not align_tar.exists() and not align_dir.exists():
        log.info("[%s] скачиваю align", spk)
        r = subprocess.run(["wget", "-q", "-O", str(align_tar), align_url])
        if r.returncode != 0:
            log.error("[%s] не удалось скачать align", spk)
            align_tar.unlink(missing_ok=True)
            return False

    # Распаковка
    if video_zip.exists():
        log.info("[%s] распаковываю видео", spk)
        subprocess.run(["unzip", "-qo", str(video_zip), "-d", str(spk_dir)],
                       check=True)
        video_zip.unlink()
    if align_tar.exists():
        log.info("[%s] распаковываю align", spk)
        subprocess.run(["tar", "-xf", str(align_tar), "-C", str(spk_dir)],
                       check=True)
        align_tar.unlink()

    return True


# =============================================================================
# Парсинг алайнов
# =============================================================================

def parse_align(align_path: Path) -> Optional[str]:
    """GRID .align → строка слов без 'sil'."""
    words = []
    try:
        with open(align_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[2] not in {"sil", "sp"}:
                    words.append(parts[2])
    except Exception as e:
        log.warning("Не могу прочитать %s: %s", align_path, e)
        return None
    return " ".join(words) if words else None


# =============================================================================
# Извлечение аудио
# =============================================================================

def extract_audio(mpg_path: Path, wav_path: Path) -> Optional[float]:
    """ffmpeg .mpg → 16 кГц моно wav. Возвращает длительность в сек или None."""
    if not wav_path.exists():
        r = subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(mpg_path), "-ar", "16000", "-ac", "1", str(wav_path),
        ])
        if r.returncode != 0:
            log.warning("ffmpeg упал на %s", mpg_path)
            return None
    try:
        import soundfile as sf
        return sf.info(str(wav_path)).duration
    except Exception as e:
        log.warning("Не могу прочитать длительность %s: %s", wav_path, e)
        return None


# =============================================================================
# Один спикер: видео → wav + .npy + записи
# =============================================================================

def _find_videos_and_aligns(spk_raw: Path) -> tuple[list[Path], dict[str, Path]]:
    """
    Рекурсивный поиск .mpg и .align внутри spk_raw, независимо от структуры
    распакованного архива (s1/s1/*.mpg vs s1/*.mpg, align/ vs s1/align/, ...).
    """
    videos = sorted(spk_raw.rglob("*.mpg"))
    aligns = {p.stem: p for p in spk_raw.rglob("*.align")}
    return videos, aligns


def process_speaker(spk: str, raw_dir: Path, processed_dir: Path,
                    lip_ext: LipROIExtractor,
                    detect_every: int = 1) -> list[dict]:
    """Обрабатывает все видео одного спикера. Возвращает записи манифеста."""
    spk_raw = raw_dir / spk
    spk_proc = processed_dir / spk
    wav_dir = spk_proc / "wav"
    lips_dir = spk_proc / "lips"
    wav_dir.mkdir(parents=True, exist_ok=True)
    lips_dir.mkdir(parents=True, exist_ok=True)

    video_files, align_map = _find_videos_and_aligns(spk_raw)
    log.info("[%s] найдено: %d видео, %d алайнов",
             spk, len(video_files), len(align_map))

    records = []
    for mpg in tqdm(video_files, desc=f"{spk} preprocess", leave=False):
        stem = mpg.stem
        align_path = align_map.get(stem)
        if align_path is None:
            continue

        text = parse_align(align_path)
        if not text:
            continue

        wav_path = wav_dir / f"{stem}.wav"
        lip_npy = lips_dir / f"{stem}.npy"

        # Аудио
        duration = extract_audio(mpg, wav_path)
        if duration is None:
            continue

        # Кропы губ
        if not lip_npy.exists():
            try:
                tensor, stats = video_to_lip_tensor(
                    mpg, extractor=lip_ext, return_stats=True,
                    detect_every=detect_every)
                arr = (tensor.squeeze(1).numpy() * 255).astype(np.uint8)
                np.save(str(lip_npy), arr)
            except Exception as e:
                log.warning("[%s/%s] губы не найдены: %s", spk, stem, e)
                continue

        records.append({
            "id": f"{spk}_{stem}",
            "speaker": spk,
            "audio": str(wav_path),
            "video": str(mpg),
            "lip_npy": str(lip_npy),
            "text": text,
            "duration": round(duration, 3),
        })

    log.info("[%s] записей: %d", spk, len(records))
    return records


# =============================================================================
# Манифесты (speaker-disjoint train/val)
# =============================================================================

def write_manifests(records: list[dict], speakers: list[str],
                    manifests_dir: Path, val_speakers: Optional[list[str]],
                    val_ratio: float = 0.1, seed: int = 42) -> None:
    """
    Разбиение train/val.

    Если val_speakers задан — спикеры из этого списка идут целиком в val
    (speaker-disjoint, более честный сплит).
    Иначе — случайный пример-уровневый сплит с указанной долей.
    """
    manifests_dir.mkdir(parents=True, exist_ok=True)
    train_path = manifests_dir / "train.jsonl"
    val_path = manifests_dir / "val.jsonl"

    if val_speakers:
        val_spk = set(val_speakers)
        train_recs = [r for r in records if r["speaker"] not in val_spk]
        val_recs = [r for r in records if r["speaker"] in val_spk]
        split_kind = f"speaker-disjoint (val={','.join(sorted(val_spk))})"
    else:
        rng = random.Random(seed)
        shuffled = records[:]
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * (1 - val_ratio))
        train_recs, val_recs = shuffled[:cut], shuffled[cut:]
        split_kind = f"random {1-val_ratio:.0%}/{val_ratio:.0%}"

    with open(train_path, "w") as f:
        for r in train_recs:
            f.write(json.dumps(r) + "\n")
    with open(val_path, "w") as f:
        for r in val_recs:
            f.write(json.dumps(r) + "\n")

    train_dur = sum(r["duration"] for r in train_recs) / 60
    val_dur = sum(r["duration"] for r in val_recs) / 60
    log.info("Разбиение: %s", split_kind)
    log.info("Манифесты: train=%d (%.1f мин), val=%d (%.1f мин)",
             len(train_recs), train_dur, len(val_recs), val_dur)


# =============================================================================
# main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speakers", default="smoke",
                        help="'smoke' / 'all10' / 'all20' / 'all', "
                             "либо список через запятую: 's1,s2,s5'")
    parser.add_argument("--base", required=True, type=Path,
                        help="корневая папка для raw/, processed/, manifests/")
    parser.add_argument("--val-speakers", default=None,
                        help="спикеры в val (через запятую). "
                             "Если не задано — случайный сплит 90/10.")
    parser.add_argument("--val-ratio", default=0.1, type=float,
                        help="доля val при случайном сплите")
    parser.add_argument("--skip-download", action="store_true",
                        help="не качать (если архивы уже распакованы)")
    parser.add_argument("--detect-every", default=5, type=int,
                        help="детекция MediaPipe на каждом N-м кадре, "
                             "рамки между ними интерполируются (~N-кратное "
                             "ускорение). 1 = на каждом кадре. По умолчанию 5 "
                             "(безопасно для GRID: диктор статичен)")
    parser.add_argument("--delegate", default="cpu", choices=["cpu", "gpu"],
                        help="делегат MediaPipe. 'gpu' пробует OpenGL ES/EGL "
                             "и при ошибке сам откатывается на CPU")
    args = parser.parse_args()

    # Парсим speakers
    if args.speakers in PRESETS:
        speakers = PRESETS[args.speakers]
    else:
        speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]

    log.info("Спикеры (%d): %s", len(speakers), ", ".join(speakers))

    val_speakers = None
    if args.val_speakers:
        val_speakers = [s.strip() for s in args.val_speakers.split(",")]
        for vs in val_speakers:
            if vs not in speakers:
                log.warning("val-спикер %s не в списке speakers — добавляю", vs)
                speakers.append(vs)

    raw_dir = args.base / "raw"
    processed_dir = args.base / "processed"
    manifests_dir = args.base / "manifests"

    # ── 1. Скачивание ──────────────────────────────────────────────────
    if not args.skip_download:
        for spk in speakers:
            ok = download_speaker(spk, raw_dir)
            if not ok:
                log.error("[%s] скачивание не удалось — пропускаю", spk)

    # ── 2. Препроцессинг ───────────────────────────────────────────────
    all_records: list[dict] = []
    log.info("MediaPipe: delegate=%s, detect_every=%d",
             args.delegate, args.detect_every)
    with LipROIExtractor(delegate=args.delegate) as lip_ext:
        for spk in speakers:
            spk_raw = raw_dir / spk
            # Ищем хотя бы один .mpg рекурсивно (структура архива может
            # отличаться: s1/s1/*.mpg или s1/*.mpg)
            has_videos = any(spk_raw.rglob("*.mpg")) if spk_raw.exists() else False
            if not has_videos:
                log.warning("[%s] не найдено .mpg в %s — пропуск", spk, spk_raw)
                continue
            recs = process_speaker(spk, raw_dir, processed_dir, lip_ext,
                                   detect_every=args.detect_every)
            all_records.extend(recs)

    log.info("Всего обработано: %d записей", len(all_records))

    # ── 3. Манифесты ───────────────────────────────────────────────────
    write_manifests(all_records, speakers, manifests_dir,
                    val_speakers=val_speakers, val_ratio=args.val_ratio)

    log.info("✅ Готово. Манифесты: %s", manifests_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
