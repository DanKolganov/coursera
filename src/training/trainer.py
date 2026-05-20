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

# TODO(day-13): импорты

# TODO(day-13): class Trainer
#   __init__(model, train_loader, val_loader, optimizer, scheduler,
#            loss_fn, device, cfg, logger)
#
#   train_one_epoch(epoch) -> dict_of_metrics
#
#   @torch.no_grad()
#   validate(epoch) -> dict_of_metrics  (loss, WER, CER)
#
#   fit(max_epochs, patience=5)
#     — каждую эпоху train + val, сохраняем best по WER, ранняя остановка
#
#   save_checkpoint(path) / load_checkpoint(path)
#     обязательно сохраняем: model_state, optimizer_state, scheduler_state,
#     epoch, best_wer  — чтобы возобновиться после отключения Colab.
