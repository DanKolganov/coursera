"""
Headless-пайплайн экспериментов GRID для главы 4.5 курсовой.

Обучает три модели на ОДНОМ backbone (замороженный Whisper-энкодер + CTC):
  - audio_only      — только аудио;
  - av_concat       — аудио+видео, слияние конкатенацией;
  - av_cross        — аудио+видео, cross-attention слияние.
Затем считает WER/CER каждой модели при разных SNR (устойчивость к шуму)
и пишет результаты в JSON-файлы.

Зачем отдельный скрипт (а не ячейки ноутбука): Colab-рантайм постоянно
отваливается, а вывод в stdout буферизуется. Поэтому здесь:
  * РЕЗУЛЬТАТЫ пишутся в файлы results/<name>.json (не в stdout) —
    читаются надёжно даже после обрыва соединения;
  * обучение РЕЗЮМИРУЕМО: чекпоинт last.pt сохраняется каждую эпоху,
    при повторном запуске обучение продолжается с места;
  * уже посчитанные модели пропускаются (есть results/<name>.json).

Запуск (в фоне, переживает дисконнекты):
    cd /content/coursera && PYTHONPATH=. nohup python -u \
        scripts/run_grid_experiments.py --base /content/grid \
        > /content/exp.log 2>&1 &

Прогресс смотреть: tail -f /content/exp.log  и  cat results/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

from src.utils.config import load_config, save_config
from src.utils.logging import get_logger
from src.data.dataset import AVSRDataset, CharTokenizer
from src.data.collate import avsr_collate_fn
from src.data.preprocessing import waveform_to_mel
from src.models.avsr_model import build_model
from src.training.trainer import Trainer
from src.training.metrics import decode_batch, wer as calc_wer, cer as calc_cer

log = get_logger("avsr.exp")

# Три эксперимента для таблицы 4.5. Все на замороженном Whisper-энкодере.
EXPERIMENTS = {
    "audio_only": {"mode": "audio_only", "fusion": "cross_attention"},
    "av_concat":  {"mode": "av",         "fusion": "concat"},
    "av_cross":   {"mode": "av",         "fusion": "cross_attention"},
}
# None = чистый звук (clean); остальные — SNR в дБ.
SNRS = [None, 20, 15, 10, 5, 0]


# ─────────────────────────────────────────────────────────────────────────────
# Конфиг для одного эксперимента
# ─────────────────────────────────────────────────────────────────────────────
def make_cfg(base_cfg_path: Path, exp: dict, train_manifest: str,
             val_manifest: str, output_dir: str, args) -> object:
    cfg = load_config(base_cfg_path)
    cfg.model.mode = exp["mode"]
    cfg.model.fusion.type = exp["fusion"]
    cfg.model.audio_encoder.freeze = True       # backbone заморожен у всех
    cfg.data.train_manifest = train_manifest
    cfg.data.val_manifest = val_manifest
    cfg.data.batch_size = args.batch_size
    cfg.data.grad_accum = 1
    cfg.data.num_workers = 2
    cfg.experiment.output_dir = output_dir
    cfg.training.lr = args.lr
    cfg.training.warmup_steps = args.warmup
    cfg.training.max_epochs = args.epochs
    cfg.training.patience = args.patience
    cfg.training.save_every_n_epochs = 1        # резюм каждую эпоху
    cfg.training.mixed_precision = False        # fp16 даёт NaN в Whisper
    return cfg


def build_loaders(cfg, tokenizer, device):
    load_video = (cfg.model.mode != "audio_only")
    train_ds = AVSRDataset(cfg.data.train_manifest, tokenizer,
                           max_duration=float(cfg.data.get("max_duration", 8.0)),
                           min_duration=float(cfg.data.get("min_duration", 0.2)),
                           load_video=load_video)
    val_ds = AVSRDataset(cfg.data.val_manifest, tokenizer,
                         max_duration=float(cfg.data.get("max_duration", 8.0)),
                         min_duration=float(cfg.data.get("min_duration", 0.2)),
                         load_video=load_video)
    pin = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=int(cfg.data.batch_size),
                              shuffle=True, num_workers=int(cfg.data.num_workers),
                              collate_fn=avsr_collate_fn, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.data.batch_size),
                            shuffle=False, num_workers=int(cfg.data.num_workers),
                            collate_fn=avsr_collate_fn, pin_memory=pin)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Оценка по SNR (шум добавляется к waveform до мел-спектрограммы)
# ─────────────────────────────────────────────────────────────────────────────
def add_noise(wav: torch.Tensor, snr_db) -> torch.Tensor:
    if snr_db is None:
        return wav
    p_sig = wav.pow(2).mean()
    p_noise = p_sig / (10 ** (snr_db / 10.0))
    return wav + torch.randn_like(wav) * p_noise.sqrt()


def load_eval_sample(entry, tokenizer, snr, load_video):
    wav, sr = torchaudio.load(entry["audio"])
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    wav = add_noise(wav, snr)
    mel = waveform_to_mel(wav, sample_rate=sr)
    if load_video and entry.get("lip_npy") and Path(entry["lip_npy"]).exists():
        arr = np.load(entry["lip_npy"])
        v = torch.from_numpy(arr).float()
        if v.ndim == 3:
            v = v.unsqueeze(1)
        if v.max() > 1.5:
            v = v / 255.0
    else:
        v = torch.zeros(1, 1, 96, 96)
    return {"id": str(entry.get("id", "")), "audio_mel": mel, "video": v,
            "text_ids": torch.tensor(tokenizer.encode(entry["text"]), dtype=torch.long),
            "text": entry["text"]}


@torch.no_grad()
def eval_by_snr(model, entries, tokenizer, device, load_video, batch_size, limit):
    model.eval()
    ents = entries[:limit] if limit else entries
    out = {}
    for snr in SNRS:
        preds, refs, buf = [], [], []

        def flush():
            if not buf:
                return
            batch = avsr_collate_fn(buf)
            logits, out_lens = model(batch["audio_mel"].to(device),
                                     batch["audio_lens"].to(device),
                                     batch["video"].to(device),
                                     batch["video_lens"].to(device))
            preds.extend(decode_batch(logits.cpu(), out_lens.cpu(), tokenizer))
            refs.extend([tokenizer.normalize(t) for t in batch["texts"]])

        for e in ents:
            buf.append(load_eval_sample(e, tokenizer, snr, load_video))
            if len(buf) >= batch_size:
                flush()
                buf = []
        flush()
        tag = "clean" if snr is None else f"{snr}dB"
        out[tag] = {"wer": round(100 * calc_wer(preds, refs), 2),
                    "cer": round(100 * calc_cer(preds, refs), 2), "n": len(refs)}
        log.info("  [%s] SNR=%s  WER=%.2f  CER=%.2f",
                 "eval", tag, out[tag]["wer"], out[tag]["cer"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def write_subset(src: Path, dst: Path, n: int):
    lines = src.read_text().splitlines()[:n]
    dst.write_text("\n".join(lines) + "\n")
    return len(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("/content/grid"))
    p.add_argument("--repo", type=Path, default=Path("/content/coursera"))
    p.add_argument("--results", type=Path, default=Path("/content/coursera/results"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--val-monitor", type=int, default=150,
                   help="сколько val-примеров для мониторинга во время обучения")
    p.add_argument("--eval-limit", type=int, default=250,
                   help="сколько val-примеров для финальной оценки по SNR")
    p.add_argument("--only", default=None, help="запустить только один эксперимент")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    args.results.mkdir(parents=True, exist_ok=True)

    man = args.base / "manifests"
    # компактный val для мониторинга во время обучения (полная оценка — в конце)
    val_small = man / "val_monitor.jsonl"
    write_subset(man / "val.jsonl", val_small, args.val_monitor)
    val_entries = [json.loads(l) for l in (man / "val.jsonl").read_text().splitlines()]
    base_cfg = args.repo / "configs" / "grid_audio_only.yaml"
    tokenizer = CharTokenizer()

    names = [args.only] if args.only else list(EXPERIMENTS)
    for name in names:
        exp = EXPERIMENTS[name]
        res_file = args.results / f"{name}.json"
        if res_file.exists():
            log.info("[%s] уже готов (%s) — пропускаю.", name, res_file)
            continue

        log.info("=== Эксперимент %s (mode=%s, fusion=%s) ===",
                 name, exp["mode"], exp["fusion"])
        out_dir = f"/content/checkpoints/{name}"
        cfg = make_cfg(base_cfg, exp, str(man / "train.jsonl"),
                       str(val_small), out_dir, args)

        train_loader, val_loader = build_loaders(cfg, tokenizer, device)
        model = build_model(cfg, vocab_size=tokenizer.vocab_size)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        save_config(cfg, Path(out_dir) / "config.yaml")
        trainer = Trainer(model=model, train_loader=train_loader,
                          val_loader=val_loader, tokenizer=tokenizer,
                          cfg=cfg, device=device, output_dir=Path(out_dir))

        last = Path(out_dir) / "last.pt"
        if last.exists():
            log.info("[%s] резюм с %s", name, last)
            trainer.load_checkpoint(last)

        trainer.fit()
        log.info("[%s] обучение завершено, считаю SNR-оценку...", name)

        snr_res = eval_by_snr(model, val_entries, tokenizer, device,
                              load_video=(exp["mode"] != "audio_only"),
                              batch_size=args.batch_size, limit=args.eval_limit)
        payload = {"name": name, "mode": exp["mode"], "fusion": exp["fusion"],
                   "best_wer_monitor": round(100 * trainer.best_wer, 2),
                   "snr": snr_res}
        res_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        log.info("[%s] РЕЗУЛЬТАТ записан → %s", name, res_file)

    # сводка
    summary = {}
    for name in EXPERIMENTS:
        f = args.results / f"{name}.json"
        if f.exists():
            summary[name] = json.loads(f.read_text())["snr"]
    (args.results / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    log.info("ГОТОВО. Сводка → %s", args.results / "summary.json")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
