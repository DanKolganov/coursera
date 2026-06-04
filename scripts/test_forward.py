"""
Smoke-test полного forward+backward pass AVSRModel.

Запуск:
    python scripts/test_forward.py \
        --video sample.mp4 \
        --text "hello world this is a test"

Что делает:
  1) Собирает мини-датасет из 2 копий одного примера.
  2) Строит полную AVSRModel из avsr_baseline.yaml.
  3) Прогоняет один forward pass — печатает размерности на каждом этапе.
  4) Считает CTC loss.
  5) Делает backward + один optimizer.step.
  6) Прогоняет ещё один forward — убеждается, что loss изменился
     (т.е. градиенты протекли).

Если всё прошло — пайплайн корректен и можно запускать настоящее обучение.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

# импорты пути
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.collate import avsr_collate_fn                  # noqa: E402
from src.data.dataset import AVSRDataset, CharTokenizer       # noqa: E402
from src.data.preprocessing import LipROIExtractor, video_to_lip_tensor  # noqa: E402
from src.models.avsr_model import build_model                 # noqa: E402
from src.training.loss import CTCLossWrapper                  # noqa: E402
from src.training.metrics import decode_batch                 # noqa: E402
from src.utils.config import load_config                      # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("smoke")


def prepare_mini_data(video_path: Path, text: str, workdir: Path,
                      n_copies: int = 2) -> Path:
    """Кешируем губы + делаем мини-манифест."""
    workdir.mkdir(parents=True, exist_ok=True)

    # Аудио — извлекаем из видео в .wav, если ещё нет
    import torchaudio
    from src.data.preprocessing import SAMPLE_RATE
    wav_path = workdir / (video_path.stem + ".wav")
    if not wav_path.exists():
        wav, sr = torchaudio.load(str(video_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE
        torchaudio.save(str(wav_path), wav, sr)

    # Губы — один раз, кешируем
    lip_npy = workdir / (video_path.stem + "_lips.npy")
    if not lip_npy.exists():
        with LipROIExtractor() as ext:
            tensor, stats = video_to_lip_tensor(
                video_path, extractor=ext, return_stats=True
            )
        arr = (tensor.squeeze(1).numpy() * 255).astype(np.uint8)
        np.save(str(lip_npy), arr)
        log.info("Cached lips: %d кадров", arr.shape[0])

    # Длительность
    import soundfile as sf
    duration = sf.info(str(wav_path)).duration

    # Манифест
    manifest = workdir / "smoke_manifest.jsonl"
    lines = []
    for k in range(n_copies):
        entry = {
            "id": f"smoke_{k:02d}",
            "audio": str(wav_path.resolve()),
            "video": str(video_path.resolve()),
            "lip_npy": str(lip_npy.resolve()),
            "text": text,
            "duration": duration,
        }
        lines.append(json.dumps(entry, ensure_ascii=False))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--text", type=str, default="hello world this is a test of avsr")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/avsr_baseline.yaml"))
    parser.add_argument("--workdir", type=Path,
                        default=Path("data/processed/_smoke"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None,
                        help="override cfg.model.mode для теста (av/audio_only/video_only)")
    args = parser.parse_args()

    if not args.video.exists():
        log.error("Видео не найдено: %s", args.video)
        return 1

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    # ── Подготовка данных ─────────────────────────────────────────────────
    manifest = prepare_mini_data(args.video, args.text, args.workdir, n_copies=2)
    log.info("Мини-манифест: %s", manifest)

    # ── Конфиг ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.mode is not None:
        cfg.model.mode = args.mode
        log.info("Override mode: %s", args.mode)
    log.info("Mode: %s, fusion: %s",
             cfg.model.mode, cfg.model.fusion.get("type", "?"))

    # ── Токенизатор + датасет + loader ────────────────────────────────────
    tokenizer = CharTokenizer()
    log.info("Vocab size: %d", tokenizer.vocab_size)

    ds = AVSRDataset(
        manifest_path=manifest,
        tokenizer=tokenizer,
        max_duration=60.0,
        min_duration=0.1,
        load_video=(cfg.model.mode != "audio_only"),
    )
    loader = DataLoader(
        ds, batch_size=2, shuffle=False, num_workers=0,
        collate_fn=avsr_collate_fn,
    )
    batch = next(iter(loader))
    log.info("Batch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            log.info("  %-12s shape=%s dtype=%s",
                     k, tuple(v.shape), v.dtype)

    # ── Модель ────────────────────────────────────────────────────────────
    log.info("Строю модель...")
    model = build_model(cfg, vocab_size=tokenizer.vocab_size).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Params: %d total, %d trainable (%.1f%%)",
             n_total, n_train, 100.0 * n_train / n_total)

    # ── Forward (без train) ───────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits, out_lens = model(
            batch["audio_mel"].to(device),
            batch["audio_lens"].to(device),
            batch["video"].to(device),
            batch["video_lens"].to(device),
        )
    log.info("Forward pass OK:")
    log.info("  logits:   shape=%s, dtype=%s, диапазон=[%.2f, %.2f]",
             tuple(logits.shape), logits.dtype,
             float(logits.min()), float(logits.max()))
    log.info("  out_lens: %s", out_lens.tolist())

    # CTC требует T_out >= max(text_lens)
    max_text_len = int(batch["text_lens"].max())
    min_out_len = int(out_lens.min())
    log.info("  T_out_min=%d, max_text_len=%d → CTC %s",
             min_out_len, max_text_len,
             "OK" if min_out_len >= max_text_len else "ПРОБЛЕМА (T_out слишком мал)")

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_fn = CTCLossWrapper(blank_id=tokenizer.BLANK_ID)
    model.train()
    logits, out_lens = model(
        batch["audio_mel"].to(device),
        batch["audio_lens"].to(device),
        batch["video"].to(device),
        batch["video_lens"].to(device),
    )
    loss = loss_fn(
        logits,
        out_lens,
        batch["text_ids"].to(device),
        batch["text_lens"].to(device),
    )
    log.info("Initial loss: %.4f", loss.item())
    assert torch.isfinite(loss), "Loss is NaN/Inf — что-то не так с пайплайном"

    # ── Backward + step ───────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )
    loss.backward()

    # Проверим, что хоть какие-то градиенты ненулевые
    n_with_grad = 0
    n_zero_grad = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            if p.grad.abs().sum() > 0:
                n_with_grad += 1
            else:
                n_zero_grad += 1
    log.info("Backward OK: %d параметров с ненулевым градиентом, %d с нулём",
             n_with_grad, n_zero_grad)

    optimizer.step()
    optimizer.zero_grad()

    # ── Второй forward — loss должен измениться ───────────────────────────
    with torch.no_grad():
        logits2, out_lens2 = model(
            batch["audio_mel"].to(device),
            batch["audio_lens"].to(device),
            batch["video"].to(device),
            batch["video_lens"].to(device),
        )
        loss2 = loss_fn(
            logits2, out_lens2,
            batch["text_ids"].to(device),
            batch["text_lens"].to(device),
        )
    log.info("Loss после 1 шага optimizer: %.4f (изменился на %.4f)",
             loss2.item(), abs(loss.item() - loss2.item()))

    # ── Greedy decode (вряд ли осмысленный после одного шага) ─────────────
    pred_texts = decode_batch(logits.float(), out_lens, tokenizer)
    log.info("Greedy decode (неотобученная модель):")
    for i, (pred, ref) in enumerate(zip(pred_texts, batch["texts"])):
        log.info("  [%d] ref=%r", i, ref)
        log.info("      pred=%r", pred)

    log.info("")
    log.info("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ — пайплайн готов к обучению.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
