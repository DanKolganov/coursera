# Обучение AVSR в Colab — полная инструкция

Этот документ — пошаговая инструкция для запуска обучения в Google Colab.
Каждая нумерованная секция = одна ячейка ноутбука. Запускай их по порядку.

## Подготовка вне Colab (один раз)

1. **GitHub-репозиторий** должен содержать актуальную версию проекта.
   Перед каждой сессией обучения сделай у себя на маке:
   ```bash
   cd /Users/d3zzle/Desktop/cursach/cursera_claude
   git add -A
   git commit -m "session: ..."
   git push
   ```

2. **Google Drive**: создай папку `MyDrive/avsr/` — туда будут сохраняться
   чекпоинты, и оттуда же будет читаться датасет. На Colab диск 80 ГБ,
   но он умирает с сессией; Drive — постоянный.

3. **Colab Pro / Pro+** (рекомендуется) или Kaggle для GPU. На бесплатном
   Colab сессия живёт ~12 часов и часто отрубается.

---

## Ячейка 1: Bootstrap

```python
# === Bootstrap: всегда первой ячейкой ===
import os, sys
print("Python:", sys.version)

# Клонируем / обновляем репо
if not os.path.isdir("/content/coursera"):
    !git clone https://github.com/DanKolganov/coursera.git /content/coursera
%cd /content/coursera
!git pull --rebase

# Зависимости. На свежем Colab уже стоят torch, torchaudio, transformers,
# opencv, librosa, soundfile, numpy. Доставляем остальное:
!pip install -q --upgrade \
    "mediapipe>=0.10.18" \
    "protobuf>=5.26" \
    jiwer \
    omegaconf \
    einops \
    tensorboard

# Проверка железа
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

> ⚠️ Если `pip install mediapipe` падает с конфликтом — **Runtime → Restart
> session**, потом снова Ячейка 1.

---

## Ячейка 2: Sanity check окружения

```python
%cd /content/coursera
!python scripts/sanity_check.py
```

Все строки должны быть `[OK]`. Если `[FAIL]` на чём-то — пиши.

---

## Ячейка 3: Подключение Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")

# Создаём папки нашего проекта на Drive
import os
DRIVE_ROOT = "/content/drive/MyDrive/avsr"
os.makedirs(f"{DRIVE_ROOT}/data", exist_ok=True)
os.makedirs(f"{DRIVE_ROOT}/checkpoints", exist_ok=True)
os.makedirs(f"{DRIVE_ROOT}/manifests", exist_ok=True)
print("Drive подключён:", DRIVE_ROOT)
```

---

## Ячейка 4: Загрузка тестового видео (selfie)

Пока нет полного датасета — обучаемся **проверять пайплайн** на одном
видео. Используй то же `sample.mp4`, которое было раньше, или залей новое.

```python
# Вариант A: загрузить с компьютера
from google.colab import files
uploaded = files.upload()
import os
sample_video = next(iter(uploaded.keys()))
# переименуем, если в имени есть пробелы/скобки
if " " in sample_video or "(" in sample_video:
    os.rename(sample_video, "sample.mp4")
    sample_video = "sample.mp4"
print("Видео:", sample_video, "размер:", os.path.getsize(sample_video) // 1024, "КБ")
```

```python
# Вариант B: если уже лежит на Drive
import shutil
shutil.copy("/content/drive/MyDrive/avsr/data/sample.mp4", "sample.mp4")
```

---

## Ячейка 5: Smoke-test forward pass

**Критическая проверка.** Перед обучением убеждаемся, что модель собирается
и через неё корректно протекают данные.

```python
!python scripts/test_forward.py \
    --video sample.mp4 \
    --text "ваша транскрипция того что в видео" \
    --mode audio_only
```

В конце должно быть `✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ`. После этого протестируем
полный AV режим:

```python
!python scripts/test_forward.py \
    --video sample.mp4 \
    --text "ваша транскрипция того что в видео" \
    --mode av
```

---

## Ячейка 6: Подготовка датасета (MUAVIC)

Здесь два пути в зависимости от того, есть ли у тебя уже данные.

### Путь A: использовать только наш sample (для проверки end-to-end)

Создаём искусственный «датасет» из одного нашего sample-видео, чтобы
проверить, что обучение запускается без ошибок:

```python
import json, os
os.makedirs("data/processed/_demo", exist_ok=True)
os.makedirs("data/manifests", exist_ok=True)

# Готовим кеш губ + wav
!python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from scripts.test_forward import prepare_mini_data
prepare_mini_data(Path('sample.mp4'),
                  'ваша транскрипция',
                  Path('data/processed/_demo'),
                  n_copies=20)  # 20 копий, чтобы DataLoader не падал
import shutil
shutil.copy('data/processed/_demo/smoke_manifest.jsonl',
            'data/manifests/train.jsonl')
shutil.copy('data/processed/_demo/smoke_manifest.jsonl',
            'data/manifests/val.jsonl')
print('Manifest created (DEBUG mode)')
"
```

> ⚠️ Это **только для smoke-теста обучения**. Модель тут переобучится на
> одном примере, не давая никакой реальной сети.

### Путь B: скачиваем MUAVIC

```python
# Качаем код MUAVIC
%cd /content
!git clone https://github.com/facebookresearch/muavic.git
%cd /content/muavic
!pip install -q -r requirements.txt

# Документация: https://github.com/facebookresearch/muavic
# Скрипты внутри качают аудио из mTEDx и видео с YouTube.
# Полный английский = 430 часов. Нам нужно ~10 ч train + 1 ч val.
# !python prepare/get_data.py --lang en --split train --output_dir /content/drive/MyDrive/avsr/data/muavic
```

> Реальные команды зависят от свежей версии MUAVIC — сверяйся с их README.
> Скачивание YouTube может падать (см. наши проблемы с yt-dlp). Запасной
> план — использовать публичный TED-talks через open-source ASR датасеты.

После загрузки запускаем препроцессинг (один раз):

```python
%cd /content/coursera
!python scripts/prepare_data.py \
    --src /content/drive/MyDrive/avsr/data/muavic/en/train \
    --dst /content/drive/MyDrive/avsr/data/lips/train \
    --manifest-out /content/drive/MyDrive/avsr/manifests/train.jsonl \
    --split train \
    --num-workers 4

!python scripts/prepare_data.py \
    --src /content/drive/MyDrive/avsr/data/muavic/en/valid \
    --dst /content/drive/MyDrive/avsr/data/lips/valid \
    --manifest-out /content/drive/MyDrive/avsr/manifests/val.jsonl \
    --split val \
    --num-workers 4
```

---

## Ячейка 7: Конфиг для Colab

Перед обучением подправим пути в конфиге так, чтобы данные читались с
Drive, а чекпоинты сохранялись туда же.

```python
import shutil, os
os.makedirs("configs", exist_ok=True)

# Создаём колаб-вариант на базе baseline
src_cfg = "configs/avsr_baseline.yaml"
dst_cfg = "configs/colab_audio_only.yaml"
shutil.copy(src_cfg, dst_cfg)

# Подставляем нужные значения через простой sed
overrides = {
    "train_manifest:.*": "train_manifest: /content/drive/MyDrive/avsr/manifests/train.jsonl",
    "val_manifest:.*":   "val_manifest: /content/drive/MyDrive/avsr/manifests/val.jsonl",
    "output_dir:.*":     "output_dir: /content/drive/MyDrive/avsr/checkpoints/audio_only_v1",
    "mode:.*":           "mode: audio_only",         # стартуем с самого простого
    "max_epochs:.*":     "max_epochs: 3",             # для первого запуска
    "batch_size:.*":     "batch_size: 4",
    "grad_accum:.*":     "grad_accum: 4",
}
for pat, repl in overrides.items():
    !sed -i 's|{pat}|{repl}|' {dst_cfg}

!cat {dst_cfg}
```

---

## Ячейка 8: Запуск обучения

```python
%cd /content/coursera
!python scripts/train.py --config configs/colab_audio_only.yaml
```

Что увидишь:
- `Vocab size: 29`
- `Датасеты: train=N, val=M`
- `Параметров всего: ~XXX M (обучаемых: ~YY M, заморожено: ~ZZ M)` —
  Whisper заморожен, обучаются только video_proj (в audio-only его нет)
  и CTC-голова.
- Прогресс-бар `Epoch 000 [train]` с loss и lr.
- В конце эпохи — `WER=X.XX, CER=Y.YY` и несколько примеров предсказаний.

**Если loss падает в первые 50-100 шагов** — пайплайн жив, можно
увеличивать эпохи и/или включать AV режим (см. Ячейку 9).

**Если loss не падает или скачет в инфинитити** — пиши, дебажим.

---

## Ячейка 9: TensorBoard (опционально)

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/avsr/checkpoints/audio_only_v1/tb_logs
```

---

## Ячейка 10: AV-режим — основной эксперимент

После того как audio-only baseline запустился, делаем то же самое для
полной AV модели:

```python
import shutil
shutil.copy("configs/avsr_baseline.yaml", "configs/colab_av.yaml")
overrides = {
    "train_manifest:.*": "train_manifest: /content/drive/MyDrive/avsr/manifests/train.jsonl",
    "val_manifest:.*":   "val_manifest: /content/drive/MyDrive/avsr/manifests/val.jsonl",
    "output_dir:.*":     "output_dir: /content/drive/MyDrive/avsr/checkpoints/av_cross_attn",
    "mode:.*":           "mode: av",
    "type:.*":           "type: cross_attention",
    "max_epochs:.*":     "max_epochs: 10",
}
for pat, repl in overrides.items():
    !sed -i 's|{pat}|{repl}|' configs/colab_av.yaml

!python scripts/train.py --config configs/colab_av.yaml
```

---

## Ячейка 11: Возобновление после отключения Colab

Если сессия отвалилась — в новой сессии прогоняешь Ячейки 1-3 заново
(bootstrap + sanity + Drive), потом:

```python
!python scripts/train.py \
    --config configs/colab_av.yaml \
    --resume /content/drive/MyDrive/avsr/checkpoints/av_cross_attn/last.pt
```

Trainer прочитает чекпоинт и продолжит с той же эпохи.

---

## Ячейка 12: Оценка

После окончания обучения — прогоняем на валидации **в нескольких режимах
шума**:

```python
# Без шума
!python scripts/eval.py \
    --checkpoint /content/drive/MyDrive/avsr/checkpoints/av_cross_attn/best.pt \
    --manifest /content/drive/MyDrive/avsr/manifests/val.jsonl

# С шумом SNR = 5 дБ
!python scripts/eval.py \
    --checkpoint /content/drive/MyDrive/avsr/checkpoints/av_cross_attn/best.pt \
    --manifest /content/drive/MyDrive/avsr/manifests/val.jsonl \
    --noise-snr 5

# Аблация: тот же чекпоинт, режим audio_only
!python scripts/eval.py \
    --checkpoint /content/drive/MyDrive/avsr/checkpoints/av_cross_attn/best.pt \
    --manifest /content/drive/MyDrive/avsr/manifests/val.jsonl \
    --mode audio_only
```

Результаты складываем в `docs/REPORT.md` в раздел «Дневник экспериментов».

---

## Чеклист на каждую сессию

- [ ] Bootstrap-ячейка прошла без ошибок
- [ ] Sanity check — все `[OK]`
- [ ] Google Drive смонтирован
- [ ] (если первая сессия) Smoke-test forward пройден
- [ ] Запустил `train.py`
- [ ] Через 30 минут зашёл проверить, что loss падает
- [ ] Записал результаты в `docs/REPORT.md`

---

## Известные ограничения и обходные пути

| Проблема | Симптом | Решение |
|----------|---------|---------|
| Colab отрубается | Сессия мертва | Bootstrap → resume с last.pt |
| OOM на GPU | `CUDA out of memory` | batch_size=2, grad_accum=8 |
| MediaPipe import error | `solutions` отсутствует | Restart runtime |
| Слишком мало данных | WER не падает / падает странно | Увеличить датасет, больше эпох |
| CTC inf loss | Loss взлетает в первой эпохе | Уменьшить lr, проверить что T_out > max_text_len |
