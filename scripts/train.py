"""
Точка входа для обучения. Запуск:
    python scripts/train.py --config configs/avsr_baseline.yaml
    python scripts/train.py --config configs/avsr_baseline.yaml --resume checkpoints/last.pt

Что делает:
  1) Загружает конфиг из YAML
  2) Фиксирует random seed
  3) Строит DataLoader'ы для train/val
  4) Инициализирует модель, Trainer
  5) Опционально загружает чекпоинт для продолжения
  6) Запускает trainer.fit()
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Чтобы импорты src.* работали при запуске как `python scripts/train.py`
# (Colab не наследует sys.path родительской сессии в подпроцесс)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config, save_config
from src.utils.logging import get_logger
from src.data.dataset import AVSRDataset, CharTokenizer
from src.data.collate import avsr_collate_fn
from src.models.avsr_model import build_model
from src.training.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Обучение AVSR модели"
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Путь к YAML конфигу (например configs/avsr_baseline.yaml)"
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Путь к чекпоинту для продолжения обучения"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="'cuda', 'cpu', или 'mps'. По умолчанию — автоопределение."
    )
    args = parser.parse_args()

    # ── Конфиг ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    log = get_logger("avsr.train")
    log.info("Конфиг загружен: %s", args.config)

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info("Device: %s", device)

    # ── Seed ──────────────────────────────────────────────────────────────
    seed = int(cfg.experiment.get("seed", 42))
    set_seed(seed)
    log.info("Random seed: %d", seed)

    # ── Токенизатор ───────────────────────────────────────────────────────
    tokenizer = CharTokenizer()
    log.info("Vocab size: %d", tokenizer.vocab_size)

    # ── Датасеты ──────────────────────────────────────────────────────────
    dcfg = cfg.data
    train_dataset = AVSRDataset(
        manifest_path=dcfg.train_manifest,
        tokenizer=tokenizer,
        max_duration=float(dcfg.get("max_duration", 15.0)),
        min_duration=float(dcfg.get("min_duration", 0.5)),
        load_video=(cfg.model.mode != "audio_only"),
    )
    val_dataset = AVSRDataset(
        manifest_path=dcfg.val_manifest,
        tokenizer=tokenizer,
        max_duration=float(dcfg.get("max_duration", 15.0)),
        min_duration=float(dcfg.get("min_duration", 0.5)),
        load_video=(cfg.model.mode != "audio_only"),
    )
    log.info(
        "Датасеты: train=%d, val=%d примеров",
        len(train_dataset), len(val_dataset),
    )

    # ── DataLoader'ы ──────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(dcfg.batch_size),
        shuffle=True,
        num_workers=int(dcfg.get("num_workers", 2)),
        collate_fn=avsr_collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=True,     # CTC не любит очень маленькие последние батчи
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(dcfg.batch_size),
        shuffle=False,
        num_workers=int(dcfg.get("num_workers", 2)),
        collate_fn=avsr_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # ── Модель ────────────────────────────────────────────────────────────
    log.info("Строю модель (mode=%s)...", cfg.model.mode)
    model = build_model(cfg, vocab_size=tokenizer.vocab_size)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Параметров всего: %d (обучаемых: %d, заморожено: %d)",
        n_total, n_train, n_total - n_train,
    )

    # ── Output directory ──────────────────────────────────────────────────
    output_dir = Path(cfg.experiment.get("output_dir", "checkpoints/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Сохраняем конфиг рядом с чекпоинтами
    save_config(cfg, output_dir / "config.yaml")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    if args.resume is not None:
        log.info("Продолжаю с чекпоинта: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    # ── Запускаем обучение ────────────────────────────────────────────────
    trainer.fit()


if __name__ == "__main__":
    main()
