"""
Оценка обученной модели на валидации/тесте.

Запуск:
    # Без шума
    python scripts/eval.py --checkpoint checkpoints/best.pt \
                           --manifest data/manifests/val.jsonl

    # С добавлением белого шума (SNR 5 дБ)
    python scripts/eval.py --checkpoint checkpoints/best.pt \
                           --manifest data/manifests/val.jsonl \
                           --noise-snr 5

    # Тест на конкретном режиме (override конфига)
    python scripts/eval.py --checkpoint checkpoints/best.pt \
                           --manifest data/manifests/test.jsonl \
                           --mode audio_only

Выводит:
  - WER и CER по всему тест-сету
  - Пример предсказаний (первые N)
  - Опционально: результаты по разным SNR уровням (для таблицы в курсовой)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Чтобы импорты src.* работали при запуске как `python scripts/eval.py`
# (Colab не наследует sys.path родительской сессии в подпроцесс)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logging import get_logger
from src.data.dataset import AVSRDataset, CharTokenizer
from src.data.collate import avsr_collate_fn
from src.models.avsr_model import build_model
from src.training.metrics import wer, cer, decode_batch

log = get_logger("avsr.eval")


def add_gaussian_noise(audio_mel: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Добавляет белый гауссов шум к мел-спектрограммам с заданным SNR (дБ).

    Это упрощённая версия: шум добавляем прямо в пространстве спектрограмм
    (а не в пространстве сигнала). Для курсовой достаточно.

    Args:
        audio_mel: (B, 80, T) — лог-мел спектрограммы.
        snr_db:    желаемое соотношение сигнал/шум в дБ.
    """
    signal_power = audio_mel.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(audio_mel) * noise_power.sqrt()
    return audio_mel + noise


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    config_override: Optional[Path] = None,
    mode_override: Optional[str] = None,
) -> tuple:
    """
    Загружает модель из чекпоинта.

    Ищет конфиг в той же папке, что и чекпоинт (config.yaml).
    Если не найден — требует явного указания через config_override.
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(str(checkpoint_path), map_location=device)

    # Ищем конфиг
    config_path = config_override or checkpoint_path.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Конфиг не найден: {config_path}. "
            f"Укажите --config явно."
        )
    cfg = load_config(config_path)

    # Переопределяем режим, если нужно
    if mode_override is not None:
        cfg.model.mode = mode_override

    tokenizer = CharTokenizer()
    model = build_model(cfg, vocab_size=tokenizer.vocab_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    log.info(
        "Модель загружена: epoch=%d, best_wer=%.4f",
        ckpt.get("epoch", "?"), ckpt.get("best_wer", float("nan")),
    )
    return model, tokenizer, cfg


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    tokenizer: CharTokenizer,
    device: torch.device,
    noise_snr_db: Optional[float] = None,
    use_amp: bool = True,
    verbose_examples: int = 5,
) -> dict:
    """
    Прогоняет модель по всему data_loader и считает метрики.

    Args:
        model:           AVSRModel в режиме eval.
        data_loader:     DataLoader тестового набора.
        tokenizer:       CharTokenizer.
        device:          torch.device.
        noise_snr_db:    если задан — добавляем шум к аудио.
        use_amp:         использовать ли FP16 autocast.
        verbose_examples: сколько примеров распечатать.

    Returns:
        {"wer": float, "cer": float, "n_samples": int}
    """
    all_preds: List[str] = []
    all_refs: List[str] = []

    for batch in tqdm(data_loader, desc="Eval"):
        # Переносим на device
        audio_mel = batch["audio_mel"].to(device, non_blocking=True)
        audio_lens = batch["audio_lens"].to(device, non_blocking=True)
        video = batch["video"].to(device, non_blocking=True)
        video_lens = batch["video_lens"].to(device, non_blocking=True)

        # Добавляем шум (если нужно)
        if noise_snr_db is not None:
            audio_mel = add_gaussian_noise(audio_mel, noise_snr_db)

        with autocast(enabled=use_amp and device.type == "cuda"):
            logits, out_lens = model(audio_mel, audio_lens, video, video_lens)

        preds = decode_batch(logits.float(), out_lens, tokenizer)
        refs = [tokenizer.normalize(t) for t in batch["texts"]]

        all_preds.extend(preds)
        all_refs.extend(refs)

    # Метрики
    val_wer = wer(all_preds, all_refs)
    val_cer = cer(all_preds, all_refs)

    # Выводим примеры
    log.info(
        "WER=%.4f (%.1f%%)  CER=%.4f (%.1f%%)  n=%d",
        val_wer, val_wer * 100,
        val_cer, val_cer * 100,
        len(all_preds),
    )
    n_show = min(verbose_examples, len(all_preds))
    for i in range(n_show):
        log.info("  [%d] ref:  '%s'", i, all_refs[i])
        log.info("  [%d] pred: '%s'", i, all_preds[i])

    return {
        "wer": val_wer,
        "cer": val_cer,
        "n_samples": len(all_preds),
        "predictions": all_preds,
        "references": all_refs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка AVSR модели"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Путь к .pt чекпоинту"
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Путь к .jsonl манифесту тест-набора"
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Путь к конфигу. По умолчанию ищем config.yaml рядом с чекпоинтом."
    )
    parser.add_argument(
        "--noise-snr", type=float, default=None,
        help="Добавить белый шум с данным SNR (дБ). Можно несколько через запятую: '0,5,10'"
    )
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["av", "audio_only", "video_only"],
        help="Переопределить режим модели (полезно для ablation study)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Размер батча для инференса"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
    )
    parser.add_argument(
        "--device", type=str, default=None,
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Если задан — сохранить результаты в JSON"
    )
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    # ── Модель ────────────────────────────────────────────────────────────
    model, tokenizer, cfg = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        config_override=args.config,
        mode_override=args.mode,
    )

    # ── Датасет ───────────────────────────────────────────────────────────
    dataset = AVSRDataset(
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        max_duration=float(cfg.data.get("max_duration", 15.0)),
        load_video=(cfg.model.mode != "audio_only"),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=avsr_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    log.info("Тест-набор: %d примеров", len(dataset))

    # ── Оценка (один или несколько SNR) ───────────────────────────────────
    all_results = {}

    snr_levels: List[Optional[float]]
    if args.noise_snr is not None:
        # Поддерживаем как одно значение ("5"), так и список ("0,5,10,15,20")
        snr_levels = [float(x) for x in str(args.noise_snr).split(",")]
    else:
        snr_levels = [None]

    for snr in snr_levels:
        label = f"SNR={snr}dB" if snr is not None else "clean"
        log.info("=== Оценка: %s ===", label)
        results = evaluate(
            model=model,
            data_loader=data_loader,
            tokenizer=tokenizer,
            device=device,
            noise_snr_db=snr,
            use_amp=True,
        )
        all_results[label] = {
            "wer": results["wer"],
            "cer": results["cer"],
            "n_samples": results["n_samples"],
        }

    # ── Финальная таблица ─────────────────────────────────────────────────
    log.info("\n%s", "=" * 50)
    log.info("%-20s  %8s  %8s", "Условие", "WER", "CER")
    log.info("-" * 50)
    for label, res in all_results.items():
        log.info(
            "%-20s  %7.2f%%  %7.2f%%",
            label, res["wer"] * 100, res["cer"] * 100,
        )
    log.info("=" * 50)

    # ── Сохраняем результаты ──────────────────────────────────────────────
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        log.info("Результаты сохранены: %s", args.output_json)


if __name__ == "__main__":
    main()
