# Рабочий журнал — AVSR курсовая

Сквозной лог работы. Заполняется по ходу спринта.
Финальная версия от **9 июня 2026** — проект сконсолидирован под GRID.

---

## Текущее состояние проекта

### Структура (после чистки)

```
cursera_claude/
├── README.md                         # исходный план курсовой
├── PROJECT.md                        # карта проекта
├── requirements.txt                  # зависимости
│
├── src/                              # ИСХОДНЫЙ КОД МОДЕЛИ
│   ├── data/
│   │   ├── preprocessing.py          # waveform_to_mel + LipROIExtractor
│   │   ├── dataset.py                # AVSRDataset + CharTokenizer
│   │   └── collate.py                # avsr_collate_fn
│   ├── models/
│   │   ├── audio_encoder.py          # замороженный Whisper-small
│   │   ├── video_encoder.py          # 3D-Conv + ResNet-18 + Transformer×4
│   │   ├── fusion.py                 # Concat + CrossAttention
│   │   └── avsr_model.py             # AVSRModel + build_model
│   ├── training/
│   │   ├── loss.py                   # CTCLossWrapper + SpecAugment
│   │   ├── metrics.py                # WER/CER + greedy CTC
│   │   └── trainer.py                # тренинг с FP16 + checkpoints
│   ├── inference/
│   │   └── realtime.py               # OfflineDemo + Gradio + RTF
│   └── utils/{config,logging}.py
│
├── configs/                          # КОНФИГИ ОБУЧЕНИЯ
│   ├── avsr_baseline.yaml            # исторический шаблон
│   ├── grid_audio_only.yaml          # ← Эксперимент 1
│   └── grid_av.yaml                  # ← Эксперимент 2 (основной)
│
├── scripts/                          # ИСПОЛНЯЕМЫЕ СКРИПТЫ
│   ├── sanity_check.py               # проверка окружения
│   ├── visualize_sample.py           # визуализация одного примера
│   ├── test_dataset.py               # smoke-test датасета
│   ├── test_forward.py               # smoke-test forward+backward
│   ├── prepare_grid.py               # ← скачивание+препроцессинг GRID
│   ├── prepare_data.py               # общий обёрточный (LRS-style)
│   ├── train.py                      # точка входа обучения
│   ├── eval.py                       # оценка + аблации с шумом
│   └── build_coursework.py           # сборка .docx
│
├── notebooks/
│   └── grid_colab.md                 # ЕДИНСТВЕННАЯ актуальная инструкция
│
└── docs/
    ├── REPORT.md                     # этот файл
    ├── afouras_2018_notes.md         # конспект ключевой статьи
    └── coursework/
        ├── 01_введение.md
        ├── 02_обзор_литературы.md
        ├── 03_метод.md
        ├── 04_эксперименты.md        # обновлено под GRID
        ├── 05_заключение.md          # обновлено под GRID
        ├── 06_список_литературы.md
        └── курсовая_работа.docx      # пересобранный финальный документ
```

### Статус компонентов

| Компонент | Статус | Примечание |
|-----------|--------|-----------|
| Окружение Colab | ✅ Готово | mediapipe Tasks API, protobuf 5.26+ |
| Предобработка | ✅ Готово | Глазами проверено |
| Токенизатор | ✅ Готово | 29 токенов, юнит-тесты пройдены |
| Датасет + Collate | ✅ Готово | smoke-test пройден |
| AudioEncoder | ✅ Готово | Whisper-small, заморожен |
| VideoEncoder | ✅ Готово | 3D-conv + ResNet-18 + Transformer |
| Fusion | ✅ Готово | Concat и Cross-Attention |
| AVSRModel | ✅ Готово | 3 режима, modality dropout |
| Trainer | ✅ Готово | FP16, gradAccum, чекпоинты |
| Eval со шумом | ✅ Готово | Гауссов шум в пространстве спектрограмм |
| Realtime/Demo | ✅ Готово | OfflineDemo + Gradio |
| Курсовая .docx | ✅ Готова | 5 глав, обновлена под GRID |
| **prepare_grid.py** | ✅ Готов | 33 спикера, speaker-disjoint сплит |
| **Скачивание данных** | ⏳ Ожидает запуска | См. notebooks/grid_colab.md |
| **Обучение Эксп. 1** | ⏳ Ожидает | audio-only baseline |
| **Обучение Эксп. 2** | ⏳ Ожидает | AV с cross-attention |

---

## Принятые проектные решения

### Архитектура
- Whisper-small encoder заморожен (88 M параметров «из коробки»).
- Видео-энкодер: Conv3D(5×7×7) → ResNet-18 → 4 слоя Transformer (~15 M обуч.).
- Fusion: 2 блока двунаправленного cross-attention (~5 M).
- CTC-голова: алфавит 29 символов, потоковый инференс.
- Modality dropout = 0.1 на AV-обучении.

### Датасет (финальный выбор)
- **GRID Corpus** (Cooke et al., 2006), 33 спикера, ~33 часа.
- Спикер-disjoint сплит: train = 10 спикеров, val = 1 спикер, test = 1 спикер.
- Узкая лексика (51 слово) — осознанное методологическое решение, см. § 4.1.1.1.
- Обоснование выбора: LipNet (2016) использовал тот же датасет, что обеспечивает
  прямую преемственность.

### Что отвергнуто и почему
- **MUAVIC** — Meta убрали `download.py` из репо, скрипт сломан, восстановление
  ненадёжно (требует ручной возни с YouTube + yt-dlp).
- **LRS3-TED** — академический запрос через Oxford VGG, ждать 2–4 недели,
  не успеем.
- **LibriSpeech** — есть аудио, нет видео, не подходит для AV-сравнения.

---

## Хронология (краткая)

### Дни 1–3
Скелет проекта, базовое окружение, предобработка (`waveform_to_mel`,
`LipROIExtractor`), визуальная проверка на одном видео.

### День 4
`CharTokenizer` + `AVSRDataset` + `avsr_collate_fn` + smoke-test датасета.

### Дни 5–10
Полная реализация модели: оба энкодера, оба варианта слияния, AVSRModel
с modality dropout, CTC + SpecAugment, Trainer с FP16 и checkpoints,
скрипты train/eval/prepare_data, OfflineDemo с RTF.

### Дни 11–13
Главы 1–5 курсовой в markdown. Сборка `курсовая_работа.docx`.
Сквозной аудит кода, smoke-test forward+backward (`scripts/test_forward.py`).

### Дни 14–16 (проблемы с данными)
- MUAVIC сломался — скрипт скачивания Meta не работает.
- Попробовал LibriSpeech через HuggingFace — годится только для audio-only.
- Финальное решение: GRID Corpus.

### День 17 (сегодня, чистка и финализация)
- Удалены устаревшие ноутбуки (`muavic_colab.ipynb`, `train_colab.ipynb`,
  `01–03_*.md`), мусорные файлы `zi5zfJph`, `avsr_src.zip`, `PROGRESS.md`
  (требует ручного удаления).
- Создан единый `scripts/prepare_grid.py` — скачивание, парсинг алайнов,
  ffmpeg→wav, MediaPipe→.npy, speaker-disjoint манифесты, идемпотентность.
- Созданы `configs/grid_audio_only.yaml` и `configs/grid_av.yaml`.
- Создана единая `notebooks/grid_colab.md` — 7 ячеек от bootstrap до eval с шумом.
- Главы 4 и 5 курсовой полностью переписаны под GRID + добавлен раздел
  4.1.1.1 «Узкий словарь как осознанное методологическое решение».
- `scripts/build_coursework.py` синхронизирован с новыми главами;
  `курсовая_работа.docx` пересобран.

---

## Известные риски

| Риск | Симптом | Решение |
|------|---------|---------|
| Бесплатный Drive 15 ГБ мал для всех 33 спикеров | OSError no space | Использовать `--speakers all10` (~16 ГБ) или гибрид Drive+local |
| Colab отрубается каждые ~12 часов | сессия мертва | bootstrap → `train.py --resume last.pt` |
| Зеркало Sheffield лежит | wget timeout | wait 1-2 часа, перезапустить скрипт (он идемпотентный) |
| OOM на T4 в AV-режиме | CUDA OOM | batch_size=2, grad_accum=8 |
| MediaPipe пропускает кадры с лицом | n_missing >50% | Параметры `min_face_detection_confidence` в `LipROIExtractor` |

---

## Что осталось сделать

1. **Удалить мусор вручную на маке** (sandbox не дал permission):
   ```bash
   cd /Users/d3zzle/Desktop/cursach/cursera_claude
   rm avsr_src.zip PROGRESS.md
   ```

2. **Закоммитить и запушить** все правки:
   ```bash
   git add -A
   git commit -m "consolidation: clean repo, GRID pipeline, sync coursework"
   git push
   ```

3. **На Drive** создать папку `MyDrive/avsr/grid` (если ещё нет).

4. **В Colab** открыть `notebooks/grid_colab.md` и пройти Ячейки 1–6
   по порядку:
   - 1: bootstrap
   - 2: sanity
   - 3: скачивание + препроцессинг (SPEAKERS = `"all10"` рекомендую)
   - 4: Эксперимент 1 — audio-only baseline
   - 5: Эксперимент 2 — AV cross-attention
   - 6: Eval со шумом

5. **Прислать сводный JSON** результатов (`results/summary.json`) или просто
   таблицу. Я заполню таблицы 4.3–4.6 курсовой реальными числами и
   пересоберу `.docx`.

---

## Дневник экспериментов (заполняется по приходе логов)

### Эксперимент 1 — Audio-only baseline (GRID, 10 спикеров)
- Конфиг: `configs/grid_audio_only.yaml`
- Состояние: ⏳ ожидает запуска
- WER (clean): TBD
- WER (SNR=0 дБ): TBD

### Эксперимент 2 — AV cross-attention (GRID, 10 спикеров)
- Конфиг: `configs/grid_av.yaml`
- Состояние: ⏳ ожидает запуска
- WER (clean): TBD
- WER (SNR=0 дБ): TBD

### Эксперимент 3 — Аблация модуля слияния
- Состояние: ⏳ ожидает запуска (после Эксп. 2)

### Эксперимент 4 — Замер RTF
- Состояние: ⏳ ожидает запуска
- RTF: TBD
- Задержка: TBD
