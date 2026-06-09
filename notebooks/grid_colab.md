# GRID на Colab — пошаговая инструкция

Единственная актуальная Colab-инструкция в проекте.

## Что у нас

- **Датасет**: GRID Corpus (Cooke et al., 2006), AV.
- **Объём**: ~33 ч (33 спикера × 1000 предложений × ~3 сек).
- **Лексика**: 51 слово, фиксированная грамматика
  `command color preposition letter digit adverb`.
- **Скачивание**: с зеркала Sheffield (`https://spandh.dcs.shef.ac.uk/gridcorpus/`).

## Объём на диске (внимание)

| Что | Размер |
|-----|--------|
| Исходные `.mpg` (33 спикера × ~700 МБ) | **~23 ГБ** |
| WAV-извлечения (16 кГц моно) | ~3 ГБ |
| Кеш губ `.npy` (33 000 видео × ~75 кадров × 96² uint8) | **~22 ГБ** |
| Чекпоинты двух моделей + tb_logs | ~3 ГБ |
| **ВСЕГО** | **~51 ГБ** |

**На бесплатном Drive (15 ГБ) НЕ влезет.** Варианты:
1. Купить Drive 100 ГБ (~$2/мес).
2. Держать `.mpg` на `/content/` (Colab диск 80 ГБ), а на Drive класть только
   `processed/` (wav+lips) и чекпоинты — это ~25 ГБ.
3. Урезать до 10 спикеров (`--speakers all10`) — ~16 ГБ всего, влезет в free Drive.

В ячейках ниже выбираешь `SPEAKERS = "all10"` или `"all"` сам.

---

## Ячейка 1 — Bootstrap

```python
import os, sys, subprocess

REPO_URL = "https://github.com/DanKolganov/coursera.git"
PROJECT_DIR = "/content/coursera"

# Клон / pull
if not os.path.isdir(PROJECT_DIR):
    subprocess.run(["git", "clone", REPO_URL, PROJECT_DIR], check=True)
else:
    subprocess.run(["git", "-C", PROJECT_DIR, "pull", "--rebase"], check=False)
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Зависимости
subprocess.run([
    "pip", "install", "-q", "--upgrade",
    "mediapipe>=0.10.18", "protobuf>=5.26",
    "jiwer", "omegaconf", "einops", "tensorboard",
], check=True)
subprocess.run(["apt-get", "-qq", "install", "-y", "ffmpeg"], check=False)

# Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
DRIVE_ROOT = "/content/drive/MyDrive/avsr"
os.makedirs(f"{DRIVE_ROOT}/grid", exist_ok=True)
os.makedirs(f"{DRIVE_ROOT}/checkpoints", exist_ok=True)

import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Repo: {PROJECT_DIR}")
print(f"Drive: {DRIVE_ROOT}")
print(">>> Сейчас: Runtime → Restart session, потом Ячейку 2")
```

После — **Runtime → Restart session** (обязательно для mediapipe).

---

## Ячейка 2 — Sanity-check

```python
%cd /content/coursera
!python scripts/sanity_check.py
```

Все строки должны быть `[OK]`.

---

## Ячейка 3 — Скачивание и препроцессинг GRID

```python
# ВЫБИРАЕМ ОБЪЁМ ДАТАСЕТА
# "smoke"  — 1 спикер  (~700 МБ, для проверки пайплайна, ~15 мин)
# "all10"  — 10 спикеров (~16 ГБ, ~10 ч речи, влезает в free Drive)
# "all20"  — 20 спикеров (~30 ГБ, ~20 ч)
# "all"    — все 33     (~50 ГБ, ~33 ч, нужен Drive 100 GB)
SPEAKERS = "all10"

# Где хранить GRID. Видео можно держать на /content/ чтобы не есть Drive,
# а processed/ обязательно на Drive (там кеш губ нужно сохранить).
GRID_BASE = "/content/drive/MyDrive/avsr/grid"   # если есть место на Drive
# Или гибрид:
# GRID_BASE = "/content/grid_raw"   # исходники локально, processed/ символлинком на Drive

!python scripts/prepare_grid.py \
    --speakers {SPEAKERS} \
    --base {GRID_BASE}
```

**Время выполнения:**
- Скачивание: 2-5 мин на спикера (~5 МБ/с).
- ffmpeg извлечение wav: ~0.2 сек на видео.
- MediaPipe препроцессинг: ~25 fps → 3 сек видео = ~3 сек обработки.

Для 10 спикеров (~10 000 видео):
- Скачивание: ~30 мин.
- Препроцессинг: ~8 ч CPU. Можно запустить и пойти спать.

Скрипт **возобновляемый**: при повторном запуске пропускает уже сделанное.

---

## Ячейка 4 — Эксперимент 1: audio-only baseline

```python
# Подменим пути в конфиге на наш Drive
import yaml
with open("configs/grid_audio_only.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["train_manifest"] = f"{GRID_BASE}/manifests/train.jsonl"
cfg["data"]["val_manifest"]   = f"{GRID_BASE}/manifests/val.jsonl"
cfg["experiment"]["output_dir"] = "/content/drive/MyDrive/avsr/checkpoints/grid_audio_only"
with open("configs/grid_audio_only_run.yaml", "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)

!python scripts/train.py --config configs/grid_audio_only_run.yaml
```

**Ожидаемый прогресс:**
- На 1 спикере: WER < 5% за 3-5 эпох.
- На 10 спикерах: WER ~5-15% к концу обучения (зависит от того, насколько одинаково говорят разные спикеры).
- Время эпохи на T4: 5-20 мин в зависимости от объёма.

---

## Ячейка 5 — Эксперимент 2: AV модель с cross-attention

```python
import yaml
with open("configs/grid_av.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["train_manifest"] = f"{GRID_BASE}/manifests/train.jsonl"
cfg["data"]["val_manifest"]   = f"{GRID_BASE}/manifests/val.jsonl"
cfg["experiment"]["output_dir"] = "/content/drive/MyDrive/avsr/checkpoints/grid_av"
with open("configs/grid_av_run.yaml", "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)

!python scripts/train.py --config configs/grid_av_run.yaml
```

---

## Ячейка 6 — Замер устойчивости к шуму

После того как обе модели обучились:

```python
AUDIO_ONLY_CKPT = "/content/drive/MyDrive/avsr/checkpoints/grid_audio_only/best.pt"
AV_CKPT         = "/content/drive/MyDrive/avsr/checkpoints/grid_av/best.pt"

import subprocess, json, os

results = {}
for ckpt, name in [(AUDIO_ONLY_CKPT, "audio_only"),
                   (AV_CKPT, "av_crossattn")]:
    results[name] = {}
    for snr in [None, 20, 15, 10, 5, 0]:
        snr_str = "clean" if snr is None else f"snr{snr}"
        print(f"\n=== {name} | {snr_str} ===")
        out_json = f"{GRID_BASE}/results/{name}_{snr_str}.json"
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        cmd = ["python", "scripts/eval.py",
               "--checkpoint", ckpt,
               "--manifest", f"{GRID_BASE}/manifests/val.jsonl",
               "--output-json", out_json]
        if snr is not None:
            cmd += ["--noise-snr", str(snr)]
        subprocess.run(cmd)
        # Парсим результат
        if os.path.exists(out_json):
            with open(out_json) as f:
                r = json.load(f)
            results[name][snr_str] = {"wer": r["wer"], "cer": r["cer"]}

# Сохраним сводку
with open(f"{GRID_BASE}/results/summary.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nИтоговая сводка:")
print(json.dumps(results, indent=2))
```

Скопируй сводку сюда — я заполню таблицы курсовой.

---

## Ячейка 7 — TensorBoard (если хочется)

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/avsr/checkpoints
```

---

## Возобновление после отключения Colab

Если сессия отвалилась посреди обучения:

```python
%cd /content/coursera
# Ячейки 1, 2 — bootstrap заново.
# Потом — продолжаем обучение с last.pt:
!python scripts/train.py \
    --config configs/grid_av_run.yaml \
    --resume /content/drive/MyDrive/avsr/checkpoints/grid_av/last.pt
```

Все состояния (epoch, optimizer, scheduler, best_wer) восстановятся.
