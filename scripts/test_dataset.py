"""
Smoke-тест датасета: из одного sample.mp4 строит мини-манифест и
прогоняет через DataLoader. Проверяет, что:
  - предобработка работает (мел-спектрограмма + кропы губ);
  - токенизатор кодирует/декодирует текст корректно;
  - collate_fn собирает батч с правильным паддингом;
  - размерности после батчинга соответствуют ожиданиям.

Запуск:
    python scripts/test_dataset.py \\
        --video sample.mp4 \\
        --text "hello world this is a test of avsr"

Что увидим в логе:
    - размер vocab токенизатора,
    - проверка normalize / encode / decode,
    - размерности audio_mel, video, text_ids в собранном батче,
    - длины каждого примера до/после паддинга.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


# Чтобы импорт `from src...` работал при запуске из корня проекта
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.collate import avsr_collate_fn, make_padding_mask     # noqa: E402
from src.data.dataset import AVSRDataset, CharTokenizer             # noqa: E402
from src.data.preprocessing import (                                # noqa: E402
    LipROIExtractor,
    SAMPLE_RATE,
    video_to_lip_tensor,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("test_dataset")


def cache_lips(video_path: Path, cache_dir: Path) -> Path:
    """Запускает MediaPipe один раз, сохраняет (T, 96, 96) uint8 в .npy."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / (video_path.stem + "_lips.npy")
    if out_path.exists():
        log.info("Кеш уже есть: %s", out_path)
        return out_path

    log.info("Извлекаю кропы губ из %s ...", video_path)
    with LipROIExtractor() as ext:
        lips_tensor, stats = video_to_lip_tensor(
            video_path, extractor=ext, return_stats=True,
        )
    # (T, 1, 96, 96) float in [0,1]  -->  (T, 96, 96) uint8
    arr = (lips_tensor.squeeze(1).numpy() * 255.0).astype(np.uint8)
    np.save(out_path, arr)
    log.info("Сохранил %d кадров губ в %s (%.2f МБ). FPS=%.1f",
             arr.shape[0], out_path, out_path.stat().st_size / 1024**2,
             stats["fps"])
    return out_path


def build_mini_manifest(
    video_path: Path,
    audio_path: Path,
    lip_npy: Path,
    text: str,
    out_path: Path,
    n_copies: int = 3,
) -> Path:
    """Создаёт jsonl с N копиями одного примера — для проверки батчинга."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Длительность по аудио — самый надёжный источник
    import soundfile as sf
    info = sf.info(str(audio_path))
    duration = info.duration

    lines = []
    for k in range(n_copies):
        entry = {
            "id": f"sample_{k:02d}",
            "audio": str(audio_path.resolve()),
            "video": str(video_path.resolve()),
            "lip_npy": str(lip_npy.resolve()),
            "text": text,
            "duration": duration,
        }
        lines.append(json.dumps(entry, ensure_ascii=False))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Записал манифест %s (%d примеров)", out_path, n_copies)
    return out_path


def ensure_wav_from_video(video_path: Path, out_dir: Path) -> Path:
    """Если рядом с видео нет .wav — вытащим аудио через torchaudio."""
    wav_path = out_dir / (video_path.stem + ".wav")
    if wav_path.exists():
        return wav_path
    import torchaudio
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Извлекаю аудио из %s в %s", video_path, wav_path)
    wav, sr = torchaudio.load(str(video_path))
    # моно
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # ресэмпл до 16 кГц, чтобы вход в датасет был унифицирован
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    torchaudio.save(str(wav_path), wav, sr)
    return wav_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True,
                        help="путь к sample.mp4")
    parser.add_argument("--text", type=str, default="hello world this is a test",
                        help="эталонная транскрипция для этого примера")
    parser.add_argument("--workdir", type=Path, default=Path("data/processed/_smoketest"),
                        help="куда сложить кеш губ, wav, манифест")
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    if not args.video.exists():
        log.error("Видео не найдено: %s", args.video)
        return 1

    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Тест токенизатора ----------
    tok = CharTokenizer()
    log.info("Tokenizer: vocab_size=%d, blank_id=%d",
             tok.vocab_size, tok.BLANK_ID)
    norm = tok.normalize(args.text)
    ids = tok.encode(args.text)
    back = tok.decode(ids)
    log.info('encode("%s") -> %d токенов', args.text, len(ids))
    log.info("  normalize:  %r", norm)
    log.info("  first ids:  %s", ids[:20])
    log.info("  decode:     %r", back)
    assert back == norm, "encode/decode не симметричны для нормализованного текста!"
    log.info("  symmetric: OK")

    # ---------- 2. Подготовка аудио + кеша губ ----------
    wav_path = ensure_wav_from_video(args.video, workdir)
    lip_npy = cache_lips(args.video, workdir)
    manifest = build_mini_manifest(
        video_path=args.video,
        audio_path=wav_path,
        lip_npy=lip_npy,
        text=args.text,
        out_path=workdir / "mini_manifest.jsonl",
        n_copies=args.batch_size,
    )

    # ---------- 3. Создаём датасет ----------
    ds = AVSRDataset(
        manifest_path=manifest,
        tokenizer=tok,
        max_duration=30.0,
        min_duration=0.1,
    )
    log.info("Размер датасета: %d примеров", len(ds))

    # ---------- 4. Один пример напрямую ----------
    sample = ds[0]
    log.info("Пример [0]:")
    log.info("  id=%s, text=%r", sample["id"], sample["text"])
    log.info("  audio_mel: %s, dtype=%s, диапазон [%.2f, %.2f]",
             tuple(sample["audio_mel"].shape), sample["audio_mel"].dtype,
             float(sample["audio_mel"].min()), float(sample["audio_mel"].max()))
    log.info("  video:     %s, dtype=%s, диапазон [%.2f, %.2f]",
             tuple(sample["video"].shape), sample["video"].dtype,
             float(sample["video"].min()), float(sample["video"].max()))
    log.info("  text_ids:  %s, длина=%d",
             sample["text_ids"][:20].tolist(), len(sample["text_ids"]))

    # ---------- 5. Через DataLoader с collate_fn ----------
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,                 # 0 — проще дебажить
        collate_fn=avsr_collate_fn,
    )
    batch = next(iter(loader))
    log.info("Собранный батч:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            log.info("  %-12s shape=%s dtype=%s",
                     k, tuple(v.shape), v.dtype)
        else:
            log.info("  %-12s = %s", k, v)

    # ---------- 6. Проверка маски паддинга ----------
    mask = make_padding_mask(batch["audio_lens"])
    log.info("Маска паддинга audio: shape=%s, паддинг-токенов=%d из %d",
             tuple(mask.shape), int(mask.sum()), mask.numel())

    # ---------- 7. Sanity: совпадает ли реальная длина после паддинга ----------
    B, n_mels, T_a_max = batch["audio_mel"].shape
    for i, real_len in enumerate(batch["audio_lens"]):
        beyond = batch["audio_mel"][i, :, real_len:].abs().max().item()
        assert beyond < 1e-8, f"После реальной длины должны быть нули, " \
                              f"но max = {beyond}"
    log.info("Паддинг audio_mel заполнен нулями: OK")

    log.info("")
    log.info("Все проверки пройдены — датасет готов.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
