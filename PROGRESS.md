# AVSR Курсовая — Журнал прогресса

> **Проект:** Audio-Visual Speech Recognition (AVSR) в реальном времени  
> **Стек:** Python 3.10, PyTorch 2.x, Whisper, MediaPipe, CTC  
> **Цель:** мультимодальная модель ASR, устойчивая к шуму за счёт чтения по губам

---

## ✅ СДЕЛАНО

### Фаза 1 — Исследование (дни 1–7)
- [x] Изучены базовые архитектуры ASR: CTC, RNN-T, LAS, Conformer, Whisper, Wav2Vec2
- [x] Изучены визуальные модели: LipNet, AV-HuBERT, Auto-AVSR
- [x] Выбрана архитектура: Whisper-encoder (аудио) + 3D-Conv/ResNet-18 (видео) + CrossAttention fusion + CTC
- [x] Прочитаны ключевые статьи (см. `docs/afouras_2018_notes.md`)
- [x] Выбран датасет: LRS3 (433ч, TED-доклады) / MUAVIC (открытый)

### Фаза 2 — Данные и предобработка (дни 4–6)
- [x] `src/data/preprocessing.py` — полная реализация:
  - `waveform_to_mel()` — аудио → лог-мел спектрограмма (80 полос, 16 кГц)
  - `LipROIExtractor` — MediaPipe FaceLandmarker → кроп губ 96×96
  - `video_to_lip_tensor()` — видеофайл → тензор (T, 1, 96, 96)
- [x] `src/data/dataset.py` — `CharTokenizer` (29 токенов) + `AVSRDataset` (JSONL манифест)
- [x] `src/data/collate.py` — `avsr_collate_fn` с паддингом переменных длин
- [x] `scripts/prepare_data.py` — пайплайн MediaPipe → .npy кропы + JSONL манифест

### Фаза 3 — Архитектура модели (дни 8–10)
- [x] `src/models/audio_encoder.py` — `AudioEncoder`:
  - Whisper-small encoder (заморожен), d_audio=768
  - Паддинг мел до 3000 фреймов, обрезка выхода по реальной длине
- [x] `src/models/video_encoder.py` — `VideoEncoder`:
  - `VideoFrontend`: Conv3d(1→64, 5×7×7) → ResNet-18 (layer1-4) → AdaptiveAvgPool → (B,T,512)
  - `SinusoidalPosEncoding` + TransformerEncoder (4 слоя, 8 голов, pre-norm)
- [x] `src/models/fusion.py` — два варианта для ablation study:
  - `ConcatFusion` — интерполяция + конкат + Linear
  - `CrossAttentionFusion` — двунаправленный cross-attention (аудио↔видео) × N слоёв
- [x] `src/models/avsr_model.py` — `AVSRModel`:
  - Режимы: `av` / `audio_only` / `video_only`
  - Modality dropout (обучает модель работать при пропаже одной модальности)
  - Метод `transcribe()` для удобного инференса

### Фаза 4 — Обучение и метрики (дни 11–13)
- [x] `src/training/loss.py` — `CTCLossWrapper` + `SpecAugment`
- [x] `src/training/metrics.py` — `ctc_greedy_decode`, `wer`, `cer` (через jiwer)
- [x] `src/training/trainer.py` — полный `Trainer`:
  - Mixed precision FP16 (GradScaler)
  - Gradient accumulation (эффективный батч = batch_size × grad_accum)
  - Linear warmup + cosine decay LR schedule
  - Early stopping по WER (patience)
  - TensorBoard логирование
  - Checkpoint save/resume (для продолжения после обрыва Colab)
- [x] `src/utils/config.py`, `src/utils/logging.py`
- [x] `configs/avsr_baseline.yaml` — полный YAML конфиг

### Фаза 5 — Скрипты и инференс (дни 13–21)
- [x] `scripts/train.py` — точка входа обучения с `--resume`
- [x] `scripts/eval.py` — оценка WER/CER на разных SNR уровнях, вывод таблицы
- [x] `src/inference/realtime.py` — `OfflineDemo` (RTF замер) + Gradio веб-интерфейс

---

## ✅ Фаза 6 — Запуск обучения на Colab (ВЫПОЛНЕНО)

- [x] Создан Colab-ноутбук (`notebooks/train_colab.ipynb`) — 12 ячеек
- [x] Отказались от Google Drive (OAuth-проблемы) → локальные пути `/content/`
- [x] Загружен `avsr_src.zip` (41 KB) с исходниками `src/` + `configs/`
- [x] Синтетические данные: 400 train + 100 val (случайные синусоиды + шум)
- [x] GPU: Tesla T4, 15.6 GB VRAM, PyTorch 2.10.0+cu128
- [x] Sanity check: логиты `[4, 1500, 29]` — форма корректна ✅
- [x] Обучение прошло 5 эпох (~1 мин), чекпоинты сохранены (`best.pt`, `last.pt`)
- [x] WER=1.0 на синт. данных — ожидаемо, **весь пайплайн работает конец-в-конец!**

> **Итог Фазы 6:** код проверен на реальном T4 GPU. Для настоящего WER нужен реальный датасет (LRS3 / MUAVIC).

---

## ✅ Дополнительно (сделано)

### Документация курсовой
- [x] `docs/coursework/01_введение.md` — полная глава ✅
- [x] `docs/coursework/02_обзор_литературы.md` — полная глава ✅
- [x] `docs/coursework/03_метод.md` — полная глава ✅
- [x] `docs/coursework/04_эксперименты.md` — структура + теория, ждёт числа ✅
- [x] `docs/coursework/05_заключение.md` — шаблон, ждёт числа ✅
- [x] `docs/coursework/06_список_литературы.md` — 20 источников ✅

### Colab-ноутбуки
- [x] `notebooks/train_colab.ipynb` — обучение на синтетике / LJSpeech ✅
- [x] `notebooks/muavic_colab.ipynb` — **НОВЫЙ**: полный пайплайн MUAVIC ✅

### Инфраструктура
- [x] `avsr_src.zip` — обновлён (64 KB, src + configs + scripts)
- [x] `scripts/prepare_data.py` — добавлен режим `--manifest-in` (для MUAVIC)

---

## 🔄 В ПРОЦЕССЕ

### Фаза 7 — Обучение на реальных данных
> **Инструкция**: открой `notebooks/muavic_colab.ipynb` в Google Colab, включи T4 GPU, запускай ячейки сверху вниз.
- [ ] Скачать MUAVIC English (~3-5 ГБ) на Google Drive (ячейка 5 ноутбука)
- [ ] Запустить `prepare_data.py` → MediaPipe кропы губ + JSONL манифесты (ячейка 7)
- [ ] Обучить audio-only baseline (ячейка 11)
- [ ] Обучить AV cross-attention модель (ячейка 10)
- [ ] Запустить eval.py → таблица WER/CER по SNR уровням (ячейка 12)

---

## 📋 ПЛАН НА БУДУ��ЕЕ

### Фаза 7 — Эксперименты и ablation study
- [ ] Обучить baseline: только аудио (`mode: audio_only`)
- [ ] Обучить baseline: только видео (`mode: video_only`)
- [ ] Обучить AV-модель с `ConcatFusion`
- [ ] Обучить AV-модель с `CrossAttentionFusion` (основная модель)
- [ ] Сравнить WER на чистом аудио и с шумом (SNR 0, 5, 10, 15, 20 дБ)
- [ ] Таблица результатов для курсовой

### Фаза 8 — Анализ результатов
- [ ] Запустить `eval.py` на тест-сете для каждого варианта
- [ ] Построить графики WER vs SNR (matplotlib)
- [ ] Визуализировать attention maps из CrossAttentionFusion
- [ ] Notebook с примерами предсказаний

### Фаза 9 — Оптимизация инференса
- [ ] Замерить RTF на CPU и GPU
- [ ] Квантизация модели (INT8) для ускорения
- [ ] Проверить работу Gradio-демо на реальных видео

### Фаза 10 — Написание курсовой
- [ ] Введение: мотивация, обзор литературы
- [ ] Описание архитектуры (с диаграммами)
- [ ] Экспериментальная часть: таблицы, графики
- [ ] Заключение и выводы

---

## 📁 Структура проекта

```
cursera_claude/
├── PROGRESS.md            ← этот файл
├── README.md              ← исходный план курсовой
├── PROJECT.md             ← рабочая документация
├── requirements.txt       ← зависимости Python
├── configs/
│   └── avsr_baseline.yaml ← гиперпараметры (✅)
├── src/
│   ├── data/
│   │   ├── preprocessing.py  ← MediaPipe + мел-спектрограмма (✅)
│   │   ├── dataset.py         ← CharTokenizer + AVSRDataset (✅)
│   │   └── collate.py         ← padding + batching (✅)
│   ├── models/
│   │   ├── audio_encoder.py   ← Whisper encoder (✅)
│   │   ├── video_encoder.py   ← 3D-Conv + ResNet-18 + Transformer (✅)
│   │   ├── fusion.py          ← CrossAttention / Concat fusion (✅)
│   │   └── avsr_model.py      ← главная модель (✅)
│   ├── training/
│   │   ├── loss.py            ← CTCLoss + SpecAugment (✅)
│   │   ├── metrics.py         ← WER, CER, greedy decode (✅)
│   │   └── trainer.py         ← полный тренировочный цикл (✅)
│   ├── inference/
│   │   └── realtime.py        ← OfflineDemo + Gradio (✅)
│   └── utils/
│       ├── config.py          ← OmegaConf loader (✅)
│       └── logging.py         ← единый логгер (✅)
├── scripts/
│   ├── prepare_data.py        ← MediaPipe → .npy + manifest (✅)
│   ├── train.py               ← точка входа обучения (✅)
│   └── eval.py                ← оценка WER/CER + шум (✅)
├── notebooks/
│   └── train_colab.ipynb      ← Colab ноутбук (🔄 в процессе)
├── data/
│   ├── raw/                   ← сырой датасет (не в git)
│   ├── processed/             ← .npy кропы губ (не в git)
│   └── manifests/             ← train.jsonl, val.jsonl
└── checkpoints/               ← веса модели (не в git)
```

---

## 🛠️ Как запустить

```bash
# 1. Подготовка данных (один раз)
python scripts/prepare_data.py \
    --src data/raw/lrs3/trainval \
    --dst data/processed/lips \
    --manifest-out data/manifests/train.jsonl \
    --split train

# 2. Обучение
python scripts/train.py --config configs/avsr_baseline.yaml

# 3. Продолжение после обрыва
python scripts/train.py --config configs/avsr_baseline.yaml \
    --resume checkpoints/avsr_baseline/last.pt

# 4. Оценка (чистое аудио + шум)
python scripts/eval.py \
    --checkpoint checkpoints/avsr_baseline/best.pt \
    --manifest data/manifests/val.jsonl \
    --noise-snr 0,5,10,15,20 \
    --output-json results/eval_results.json

# 5. Градио-демо
python -m src.inference.realtime \
    --checkpoint checkpoints/avsr_baseline/best.pt \
    --gradio
```

---

*Последнее обновление: автоматически при каждом изменении*
