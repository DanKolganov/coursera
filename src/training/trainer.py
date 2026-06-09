"""
Тренировочный цикл. Намеренно пишем "руками", без PyTorch Lightning —
так понятнее, что происходит, и проще дебажить на Colab.

Структура одной эпохи:
  for batch in train_loader:
      batch = move_to_device(batch)
      with autocast():                    # mixed precision FP16
          logits, lens = model(batch)
          loss = ctc_loss(logits, lens, ...)
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      torch.nn.utils.clip_grad_norm_(..., 5.0)
      scaler.step(optimizer); scaler.update()
      scheduler.step()
      log(loss, lr)
  validate()
  save_checkpoint() if improved
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.training.loss import CTCLossWrapper, SpecAugment
from src.training.metrics import compute_metrics


log = logging.getLogger("avsr.trainer")


# =============================================================================
# Learning rate scheduler: linear warmup → cosine decay
# =============================================================================

def get_linear_warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Linear warmup за первые warmup_steps шагов,
    затем cosine decay до min_lr_ratio * base_lr.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Тренировочный цикл для AVSR.

    Ключевые возможности:
      - Mixed precision (FP16) через GradScaler
      - Gradient accumulation (эффективный батч = batch_size × grad_accum)
      - Gradient clipping
      - SpecAugment на мел-спектрограммах
      - Checkpoint save/resume (сохраняем всё для продолжения на Colab)
      - TensorBoard логирование
      - Early stopping по WER

    Args:
        model:        AVSRModel.
        train_loader: DataLoader для обучения.
        val_loader:   DataLoader для валидации.
        tokenizer:    CharTokenizer.
        cfg:          OmegaConf DictConfig (полный конфиг).
        device:       torch.device.
        output_dir:   куда сохранять чекпоинты и логи.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        cfg,
        device: torch.device,
        output_dir: str | Path,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg.training

        # ── Optimizer ─────────────────────────────────────────────────────
        # Замороженные параметры Whisper не включаем в optimizer
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable,
            lr=float(tcfg.lr),
            weight_decay=float(tcfg.weight_decay),
        )

        # ── Scheduler ─────────────────────────────────────────────────────
        # Рассчитываем total_steps для scheduler
        steps_per_epoch = math.ceil(
            len(train_loader) / int(tcfg.get("grad_accum", 1))
        )
        total_steps = steps_per_epoch * int(tcfg.max_epochs)
        self.scheduler = get_linear_warmup_cosine_schedule(
            self.optimizer,
            warmup_steps=int(tcfg.warmup_steps),
            total_steps=total_steps,
        )

        # ── Loss ──────────────────────────────────────────────────────────
        self.loss_fn = CTCLossWrapper(blank_id=self.tokenizer.BLANK_ID)

        # ── SpecAugment ───────────────────────────────────────────────────
        aug_cfg = cfg.get("augmentation", {}).get("audio", {})
        if aug_cfg.get("spec_augment", True):
            self.spec_augment = SpecAugment(
                freq_mask_param=int(aug_cfg.get("freq_mask", 27)),
                time_mask_param=int(aug_cfg.get("time_mask", 10)),
                num_freq_masks=1,
                num_time_masks=2,
            ).to(device)
        else:
            self.spec_augment = None

        # ── Mixed precision ───────────────────────────────────────────────
        self.use_amp = bool(tcfg.get("mixed_precision", True)) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # ── Gradient accumulation ─────────────────────────────────────────
        self.grad_accum: int = int(tcfg.get("grad_accum", 1))
        self.grad_clip: float = float(tcfg.get("grad_clip", 5.0))

        # ── Логирование ───────────────────────────────────────────────────
        log_cfg = cfg.get("logging", {})
        self.log_every_n_steps: int = int(log_cfg.get("log_every_n_steps", 50))
        self.val_every_n_epochs: int = int(log_cfg.get("val_every_n_epochs", 1))

        tb_dir = self.output_dir / "tb_logs"
        self.writer: Optional[SummaryWriter] = None
        if log_cfg.get("use_tensorboard", True):
            self.writer = SummaryWriter(str(tb_dir))

        # ── Состояние тренировки ──────────────────────────────────────────
        self.global_step: int = 0
        self.start_epoch: int = 0
        self.best_wer: float = float("inf")

    # =========================================================================
    # Обучение одной эпохи
    # =========================================================================

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:03d} [train]",
            leave=False,
        )

        for step, batch in enumerate(pbar):
            batch = self._to_device(batch)

            # SpecAugment на мел-спектрограммах
            if self.spec_augment is not None:
                batch["audio_mel"] = self.spec_augment(batch["audio_mel"])

            # Forward + loss
            with autocast(enabled=self.use_amp):
                logits, out_lens = self.model(
                    batch["audio_mel"],
                    batch["audio_lens"],
                    batch["video"],
                    batch["video_lens"],
                )
                loss = self.loss_fn(
                    logits,
                    out_lens,
                    batch["text_ids"],
                    batch["text_lens"],
                )
                # Нормируем на grad_accum для корректной величины градиентов
                loss_scaled = loss / self.grad_accum

            self.scaler.scale(loss_scaled).backward()

            total_loss += loss.item()
            n_batches += 1

            # Шаг оптимизатора каждые grad_accum батчей
            if (step + 1) % self.grad_accum == 0 or (step + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Логирование
                if self.global_step % self.log_every_n_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
                    if self.writer:
                        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                        self.writer.add_scalar("train/lr", lr, self.global_step)

        avg_loss = total_loss / max(n_batches, 1)
        return {"loss": avg_loss}

    # =========================================================================
    # Валидация
    # =========================================================================

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds: list[str] = []
        all_refs: list[str] = []
        n_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:03d} [ val ]",
            leave=False,
        )

        for batch in pbar:
            batch = self._to_device(batch)

            with autocast(enabled=self.use_amp):
                logits, out_lens = self.model(
                    batch["audio_mel"],
                    batch["audio_lens"],
                    batch["video"],
                    batch["video_lens"],
                )
                loss = self.loss_fn(
                    logits,
                    out_lens,
                    batch["text_ids"],
                    batch["text_lens"],
                )

            metrics = compute_metrics(
                logits.float(),
                out_lens,
                batch["texts"],
                self.tokenizer,
                blank_id=self.tokenizer.BLANK_ID,
            )

            total_loss += loss.item()
            all_preds.extend(metrics["predictions"])
            all_refs.extend([self.tokenizer.normalize(t) for t in batch["texts"]])
            n_batches += 1

        from src.training.metrics import wer as calc_wer, cer as calc_cer
        avg_loss = total_loss / max(n_batches, 1)
        val_wer = calc_wer(all_preds, all_refs)
        val_cer = calc_cer(all_preds, all_refs)

        if self.writer:
            self.writer.add_scalar("val/loss", avg_loss, epoch)
            self.writer.add_scalar("val/wer", val_wer, epoch)
            self.writer.add_scalar("val/cer", val_cer, epoch)

        # Несколько примеров для визуального контроля
        n_show = min(3, len(all_preds))
        for i in range(n_show):
            log.info(
                "  [val sample %d]  ref: '%s'  |  pred: '%s'",
                i, all_refs[i], all_preds[i],
            )

        return {"loss": avg_loss, "wer": val_wer, "cer": val_cer}

    # =========================================================================
    # Основной цикл обучения
    # =========================================================================

    def fit(
        self,
        max_epochs: Optional[int] = None,
        patience: Optional[int] = None,
    ) -> None:
        """
        Запускает обучение.

        Args:
            max_epochs: перекрывает cfg.training.max_epochs если задан.
            patience:   кол-во эпох без улучшения WER до early stopping.
        """
        tcfg = self.cfg.training
        max_epochs = max_epochs or int(tcfg.max_epochs)
        patience = patience or int(tcfg.get("patience", 5))

        epochs_no_improve = 0
        start_time = time.time()

        log.info(
            "Запускаю обучение: %d эпох, patience=%d, device=%s",
            max_epochs, patience, self.device,
        )

        for epoch in range(self.start_epoch, max_epochs):
            # --- Train ---
            train_metrics = self.train_one_epoch(epoch)

            # --- Validate (не каждую эпоху, если настроено) ---
            if (epoch + 1) % self.val_every_n_epochs == 0:
                val_metrics = self.validate(epoch)
                wer_val = val_metrics["wer"]

                elapsed = (time.time() - start_time) / 60.0
                log.info(
                    "Epoch %03d/%03d  train_loss=%.4f  val_loss=%.4f  "
                    "WER=%.4f  CER=%.4f  elapsed=%.1f мин",
                    epoch, max_epochs,
                    train_metrics["loss"], val_metrics["loss"],
                    wer_val, val_metrics["cer"],
                    elapsed,
                )

                # Сохраняем лучшую модель
                if wer_val < self.best_wer:
                    self.best_wer = wer_val
                    epochs_no_improve = 0
                    self.save_checkpoint(self.output_dir / "best.pt", epoch)
                    log.info("  ✓ Новый лучший WER=%.4f, чекпоинт сохранён.", wer_val)
                else:
                    epochs_no_improve += 1
                    log.info(
                        "  WER не улучшился (%d/%d).", epochs_no_improve, patience
                    )
                    if epochs_no_improve >= patience:
                        log.info("Early stopping.")
                        break
            else:
                log.info(
                    "Epoch %03d/%03d  train_loss=%.4f",
                    epoch, max_epochs, train_metrics["loss"],
                )

            # Сохраняем последний чекпоинт (для возобновления)
            self.save_checkpoint(self.output_dir / "last.pt", epoch)

        if self.writer:
            self.writer.close()
        log.info(
            "Обучение завершено. Лучший WER=%.4f", self.best_wer
        )

    # =========================================================================
    # Чекпоинты
    # =========================================================================

    def save_checkpoint(self, path: str | Path, epoch: int) -> None:
        """
        Сохраняет состояние: модель, оптимизатор, планировщик, метаданные.
        Это позволяет продолжить обучение после отключения Colab.
        """
        path = Path(path)
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_wer": self.best_wer,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
            },
            str(path),
        )
        log.debug("Чекпоинт сохранён: %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Загружает чекпоинт и восстанавливает все состояния.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {path}")

        ckpt = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_wer = float(ckpt.get("best_wer", float("inf")))

        log.info(
            "Загружен чекпоинт %s (epoch=%d, best_wer=%.4f)",
            path.name, self.start_epoch - 1, self.best_wer,
        )

    # =========================================================================
    # Утилита
    # =========================================================================

    def _to_device(self, batch: dict) -> dict:
        """Перемещаем тензорные поля батча на нужный device."""
        tensor_keys = {
            "audio_mel", "audio_lens", "video", "video_lens",
            "text_ids", "text_lens",
        }
        return {
            k: v.to(self.device, non_blocking=True) if k in tensor_keys and isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
