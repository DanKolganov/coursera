# One-paste обучение в Colab (минимум кликов)

Цель: запустить полный цикл (клон → проверки → smoke-test → обучение)
с минимумом ручных действий. Каждый блок — **одна ячейка Colab**,
просто копируешь и нажимаешь Shift+Enter.

> Между ячейками 1 и 2 — **обязательный рестарт сессии** (Runtime → Restart
> session), иначе mediapipe и protobuf не подцепятся правильно.

---

## Ячейка 1 — Bootstrap (один раз на сессию)

```python
# === BOOTSTRAP ===
import os, sys, subprocess

REPO_URL = "https://github.com/DanKolganov/coursera.git"
REPO_DIR = "/content/coursera"

# Клон / обновление
if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
subprocess.run(["git", "pull", "--rebase"], check=False)

# Зависимости
subprocess.run([
    "pip", "install", "-q", "--upgrade",
    "mediapipe>=0.10.18", "protobuf>=5.26",
    "jiwer", "omegaconf", "einops", "tensorboard",
], check=True)

# Drive
from google.colab import drive
drive.mount("/content/drive", force_remount=False)
DRIVE_ROOT = "/content/drive/MyDrive/avsr"
os.makedirs(f"{DRIVE_ROOT}/checkpoints", exist_ok=True)
os.makedirs(f"{DRIVE_ROOT}/data", exist_ok=True)

import torch
print("=" * 60)
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Repo: {REPO_DIR}")
print(f"Drive: {DRIVE_ROOT}")
print("=" * 60)
print(">>> ТЕПЕРЬ: Runtime → Restart session (обязательно)")
print(">>> Потом запусти Ячейку 2")
```

После этой ячейки — **жми Runtime → Restart session** (или Ctrl+M, .).

---

## Ячейка 2 — Sanity + загрузка тестового видео

После рестарта:

```python
# === SANITY + UPLOAD ===
import os, sys
os.chdir("/content/coursera")
sys.path.insert(0, "/content/coursera")

# Проверка установки
import subprocess
subprocess.run(["python", "scripts/sanity_check.py"], check=False)

# Загружаем видео (если ещё нет)
if not os.path.exists("sample.mp4"):
    from google.colab import files
    print("\n>>> Загрузи короткое видео (5-10 сек, чёткое лицо)")
    uploaded = files.upload()
    fname = next(iter(uploaded.keys()))
    if fname != "sample.mp4":
        os.rename(fname, "sample.mp4")
    print(f"Готово: sample.mp4, {os.path.getsize('sample.mp4')//1024} КБ")
else:
    print(f"sample.mp4 уже есть: {os.path.getsize('sample.mp4')//1024} КБ")
```

---

## Ячейка 3 — Smoke-test forward pass

```python
# === SMOKE TEST ===
# Замени текст на то, что реально говорится в твоём видео
SAMPLE_TEXT = "hello world this is a test"

import subprocess
print("--- AUDIO ONLY MODE ---")
r = subprocess.run([
    "python", "scripts/test_forward.py",
    "--video", "sample.mp4",
    "--text", SAMPLE_TEXT,
    "--mode", "audio_only",
])
assert r.returncode == 0, "audio_only smoke test failed"

print("\n--- AV MODE ---")
r = subprocess.run([
    "python", "scripts/test_forward.py",
    "--video", "sample.mp4",
    "--text", SAMPLE_TEXT,
    "--mode", "av",
])
assert r.returncode == 0, "av smoke test failed"

print("\n✅ Forward pass OK в обоих режимах — пайплайн жив.")
```

В конце должны увидеть две зелёные галочки. Если что-то падает — лог сюда.

---

## Ячейка 4 — Подготовка мини-датасета (20 копий sample.mp4)

Это **не настоящее обучение**, а проверка, что петля train.py крутится
и loss падает (модель оверфитнется на одном примере — это нормально).
Реальный датасет MUAVIC — отдельный этап.

```python
# === MINI DATASET ===
SAMPLE_TEXT = "hello world this is a test"   # ← синхронизируй с Ячейкой 3

import sys, os, json, shutil
os.chdir("/content/coursera")
sys.path.insert(0, "/content/coursera")

from pathlib import Path
from scripts.test_forward import prepare_mini_data

manifest = prepare_mini_data(
    Path("sample.mp4"),
    SAMPLE_TEXT,
    Path("data/processed/_demo"),
    n_copies=20,
)
os.makedirs("data/manifests", exist_ok=True)
shutil.copy(manifest, "data/manifests/train.jsonl")
shutil.copy(manifest, "data/manifests/val.jsonl")

with open("data/manifests/train.jsonl") as f:
    print(f"train.jsonl: {sum(1 for _ in f)} записей")
```

---

## Ячейка 5 — Конфиг для audio-only baseline

```python
# === CONFIG ===
import os, shutil, re
os.chdir("/content/coursera")

src = "configs/avsr_baseline.yaml"
dst = "configs/colab_audio_only_demo.yaml"
shutil.copy(src, dst)

with open(dst) as f:
    cfg = f.read()

# Подменяем параметры
overrides = {
    r"train_manifest:.*": "train_manifest: data/manifests/train.jsonl",
    r"val_manifest:.*":   "val_manifest: data/manifests/val.jsonl",
    r"output_dir:.*":     "output_dir: /content/drive/MyDrive/avsr/checkpoints/audio_only_demo",
    r"  mode:.*":         "  mode: audio_only",
    r"max_epochs:.*":     "max_epochs: 3",
    r"warmup_steps:.*":   "warmup_steps: 20",
    r"batch_size:.*":     "batch_size: 2",
    r"grad_accum:.*":     "grad_accum: 1",
    r"num_workers:.*":    "num_workers: 0",
}
for pat, repl in overrides.items():
    cfg = re.sub(pat, repl, cfg)
with open(dst, "w") as f:
    f.write(cfg)

print(f"Конфиг готов: {dst}")
print("---")
print(cfg)
```

---

## Ячейка 6 — Запуск train.py

```python
# === TRAIN ===
import subprocess, os
os.chdir("/content/coursera")
subprocess.run([
    "python", "scripts/train.py",
    "--config", "configs/colab_audio_only_demo.yaml",
])
```

**Что ждать в логах**:
- `Vocab size: 29`
- `Датасеты: train=20, val=20`
- `Параметров всего: ~XX M (обучаемых: ~Y M, заморожено: ~Z M)` — Whisper заморожен.
- Прогресс-бар `Epoch 000 [train]` с падающим loss.
- На validation — `WER=...` (на 20 одинаковых примерах должен упасть к нулю за 1-2 эпохи — модель оверфитнется).

Если loss падает за первые 30-50 шагов — пайплайн жив, можно
запускать на реальном датасете.

---

## Ячейка 7 — TensorBoard (опционально)

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/avsr/checkpoints/audio_only_demo/tb_logs
```

---

## Что копировать сюда мне

Чтобы я мог заполнить дневник экспериментов в REPORT.md и таблицы в
курсовой — копируй из логов:
- последнюю строчку `Epoch 002/3  train_loss=X.XX  val_loss=Y.YY  WER=Z.ZZ`;
- (если ошибки) — последние ~30 строк traceback'а.

Не нужны скриншоты — текст логов мне удобнее парсить.
