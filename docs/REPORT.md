# Рабочий журнал — AVSR курсовая

Сквозной лог работы над проектом. Сюда заносим **всё**: что сделано, какие
проблемы встретились, какие решения приняли, результаты экспериментов.
Этот документ — основа для главы «Эксперименты» и для устной защиты.

---

## Структура проекта (snapshot)

```
cursera_claude/
├── PROJECT.md                       # карта проекта
├── README.md                        # исходный план курсовой
├── requirements.txt                 # зависимости (mediapipe>=0.10.18, protobuf>=5.26)
├── configs/
│   └── avsr_baseline.yaml           # базовый эксперимент: AV + cross-attention
├── src/
│   ├── data/
│   │   ├── preprocessing.py         # ✅ waveform_to_mel + LipROIExtractor (Tasks API)
│   │   ├── dataset.py               # ✅ AVSRDataset + CharTokenizer (29 токенов)
│   │   └── collate.py               # ✅ avsr_collate_fn + make_padding_mask
│   ├── models/
│   │   ├── audio_encoder.py         # ✅ AudioEncoder = Whisper-small encoder (frozen)
│   │   ├── video_encoder.py         # ✅ 3D-Conv + ResNet-18 + Transformer×4
│   │   ├── fusion.py                # ✅ ConcatFusion + CrossAttentionFusion×2
│   │   └── avsr_model.py            # ✅ AVSRModel + build_model + modality_dropout
│   ├── training/
│   │   ├── loss.py                  # ✅ CTCLossWrapper + SpecAugment
│   │   ├── metrics.py               # ✅ WER/CER + ctc_greedy_decode
│   │   └── trainer.py               # ✅ Trainer (FP16, grad_accum, чекпоинты)
│   └── inference/
│       └── realtime.py              # ✅ OfflineDemo (RTF), Gradio-интерфейс
├── scripts/
│   ├── sanity_check.py              # проверка окружения
│   ├── visualize_sample.py          # визуализация мел + кропов
│   ├── test_dataset.py              # smoke-test датасета
│   ├── prepare_data.py              # ✅ препроцессинг датасета параллельно
│   ├── train.py                     # ✅ точка входа обучения
│   └── eval.py                      # ✅ оценка + аблации с шумом
└── docs/
    ├── REPORT.md                    # этот файл
    ├── afouras_2018_notes.md        # конспект ключевой статьи
    └── coursework/                  # главы курсовой (заполняются)
```

---

## Хронология

### День 1-3 (закрыты)

- Создан скелет проекта (структура `src/`, `configs/`, `scripts/`, `docs/`).
- Написан `requirements.txt`, поднята среда в Colab (T4 GPU, бесплатный
  тариф).
- Реализована предобработка: `waveform_to_mel` (80 мел-полос, hop 10 мс),
  `LipROIExtractor` через MediaPipe Tasks API (FaceLandmarker, 478 точек,
  ~40 индексов губ → bbox + 20% padding → 96×96 grayscale).
- Решены проблемы окружения:
  - `mediapipe==0.10.14` несовместим с `protobuf 5.26+` (TF в свежем
    Colab) → переход на актуальный MediaPipe + Tasks API
    (`FaceLandmarker.create_from_options(...)` вместо `FaceMesh`).
  - URL модели `face_landmarker.task` (~3 МБ), кешируется в
    `~/.cache/avsr/`.
- Визуализация одного примера через `scripts/visualize_sample.py` —
  кропы губ + лог-мел спектрограмма. На контрольном примере (видео
  3.79 сек, 115 кадров) кропы корректно покрывают губы, в спектрограмме
  видны выраженные формантные дорожки в фреймах 120-200 (гласные).

### День 4 (закрыт)

- `CharTokenizer`: vocab_size = 29 (1 blank + 26 букв + пробел +
  апостроф). Нормализация: lowercase, унификация апострофов разных видов,
  фильтр пунктуации/цифр, схлопывание пробелов. Юнит-тестами проверена
  симметрия encode/decode и корректность CTC-стиля декодирования
  (`[_,h,h,_,e,l,l,_,l,o,o,_] → "hello"`).
- `AVSRDataset`: jsonl-манифест, фильтр по длительности (0.5-15 сек),
  поддержка кеша губ из `.npy` и fallback на runtime-извлечение (только
  для отладки). Audio-only режим (`load_video=False`).
- `avsr_collate_fn`: паддинг переменных длин аудио/видео нулями, flatten
  текста для CTCLoss. Утилита `make_padding_mask` для энкодеров.
- Smoke-test `scripts/test_dataset.py` — мини-манифест из 3 копий одного
  примера, прогон через DataLoader, проверка размерностей.

### День 5-7 (модели, реализация)

- `AudioEncoder`: обёртка над Whisper-small (`openai/whisper-small`), берём
  только encoder, веса заморожены. Whisper ожидает (B, 80, 3000) — паддим
  до этого размера, потом обрезаем выход по реальной длине
  (`ceil(mel_lens / 2)` из-за двух Conv1d(stride=2) в stem). `d_audio = 768`.
- `VideoEncoder`: 3-уровневая архитектура.
  1. `VideoFrontend`: `Conv3d(1→64, kernel=(5,7,7), stride=(1,2,2))` →
     `BatchNorm3d` → `ReLU` → `MaxPool3d` → ResNet-18 (`layer1..layer4`,
     `AdaptiveAvgPool2d(1)`). Выход: вектор 512 на каждый кадр.
  2. Linear-проекция 512 → d_model + синусоидальное positional encoding.
  3. `TransformerEncoder` (4 слоя, 8 голов, pre-norm).
  - Маска паддинга на основе `video_lens` для корректного self-attention.
- `Fusion` — два варианта:
  - `ConcatFusion`: интерполяция видео до длины аудио → конкат → Linear.
  - `CrossAttentionFusion`: 2 блока двунаправленного cross-attention
    (audio→video и video→audio в pre-norm стиле, GELU FFN), затем
    интерполяция + конкат + LayerNorm.
- `AVSRModel`: три режима (`av` / `audio_only` / `video_only`),
  `modality_dropout=0.1` (с равной вероятностью обнуляем аудио или видео),
  CTC-голова `Linear(d_model, vocab_size)` с init `N(0, 0.02)`.

### День 8-11 (тренировка, метрики, скрипты)

- `CTCLossWrapper`: log_softmax + transpose (B,T,V)→(T,B,V) +
  `zero_infinity=True` для стабильности.
- `SpecAugment`: 2 временных маски, 1 частотная, активна только в train.
- `metrics`: `ctc_greedy_decode` + `wer` + `cer` через `jiwer`.
- `Trainer`: FP16 autocast + GradScaler, gradient accumulation
  (эффективный батч = 4×4=16), gradient clipping (5.0), linear warmup +
  cosine decay LR scheduler, TensorBoard логирование, чекпоинты с полным
  состоянием для возобновления (model + optimizer + scheduler + scaler).
- `scripts/train.py`: загрузка YAML конфига, фиксация seed, построение
  датасетов и модели, фабрика Trainer, поддержка `--resume`.
- `scripts/eval.py`: оценка с опциональным наложением шума (белый гауссов
  в пространстве спектрограмм с заданным SNR в дБ), переопределение
  режима модели (`--mode audio_only/video_only/av`) — для аблаций.
- `scripts/prepare_data.py`: параллельный препроцессинг датасета,
  многопроцессорный режим через `ProcessPoolExecutor`, сборка манифеста.

### День 14 (текущий)

- ✅ Подготовлен `notebooks/03_one_paste_train.md` — компактный
  ready-to-paste сценарий для запуска обучения в Colab из 7 ячеек:
  bootstrap → sanity → upload → smoke-test → mini-dataset → config →
  train. Минимум кликов со стороны пользователя.
- Проверено состояние десктопа: Colab открыт в Chrome (вкладка
  Untitled0.ipynb), готов к запуску.
- **Ожидается**: запуск Ячеек 1-6 в Colab, передача логов для
  заполнения дневника экспериментов.

### День 12-13 (спринт-режим)

- Аудит кода завершён: все 12 файлов реализации прочитаны и
  верифицированы (`audio_encoder`, `video_encoder`, `fusion`,
  `avsr_model`, `loss`, `metrics`, `trainer`, `train.py`, `eval.py`,
  `prepare_data.py`, `realtime.py`). Реализация полная и согласованная.
- Написан `scripts/test_forward.py` — smoke-test полного forward+backward
  pass с проверкой размерностей, ненулевых градиентов и снижения loss
  после одного шага оптимизатора.
- Создана исчерпывающая инструкция `notebooks/02_train_in_colab.md`:
  bootstrap, Drive, smoke-test, запуск train.py, resume, eval.
- Написаны все 5 глав курсовой в markdown (`docs/coursework/01..05`).
- ✅ **Собрана курсовая работа в формате .docx**: `docs/coursework/
  курсовая_работа.docx` (~66 КБ, ~45 000 символов, 328 параграфов,
  9 таблиц, иерархия H1/H2/H3 для автогенерации содержания).
  Форматирование: Times New Roman 14pt, полуторный интервал,
  поля 30/15/20/20 мм, А4, отступ первой строки 1.25 см.
  Заглушки: название университета, кафедры, ФИО руководителя
  (под замену перед сдачей).

---

## Статус компонентов

| Компонент | Статус | Примечание |
|-----------|--------|------------|
| Окружение Colab | ✅ Работает | mediapipe Tasks API, protobuf 5.26+ |
| Предобработка | ✅ Работает | Визуально проверено на 1 примере |
| Токенизатор | ✅ Работает | 29 токенов, юнит-тесты пройдены |
| Датасет + Collate | ✅ Работает | smoke-test пройден |
| AudioEncoder | ✅ Реализован | Whisper-small, заморожен. Готов к smoke-тесту в Colab |
| VideoEncoder | ✅ Реализован | 3D-conv + ResNet-18 + Transformer |
| Fusion | ✅ Реализован | Concat и Cross-Attention оба готовы |
| AVSRModel (полная) | ✅ Реализован | Сборка + modality dropout + 3 режима |
| Trainer | ✅ Реализован | FP16, gradAccum, ckpts, ранний останов |
| Eval со шумом | ✅ Реализован | Гауссов шум в пространстве спектрограмм |
| Realtime/Demo | ✅ Реализован | OfflineDemo + Gradio + замер RTF |
| **Курсовая работа .docx** | ✅ Собрана | 5 глав, 9 таблиц, источники |
| **Данные MUAVIC** | ⏳ Не загружены | Зависит от запуска в Colab |
| **Обучение** | ⏳ Не запущено | Ожидает запуска в Colab по 02_train_in_colab.md |

---

## Известные риски и решения

1. **Бесплатный Colab отключает сессию каждые ~12 часов**.
   Решение: чекпоинты на Google Drive каждые 30 минут, поддержка
   `--resume` в `train.py`.
2. **MediaPipe на CPU медленный** (~30 кадров/сек). При препроцессинге
   10 часов видео — ~3 часа CPU-времени. Решение: параллельный режим
   `--num-workers 4` (на Colab Hyper Threading реально даёт x2-3).
3. **Несовпадение длин аудио и видео**. Решение: интерполяция видео до
   длины аудио в Fusion (выход модели согласован с аудио-временной осью).
4. **Whisper-encoder ждёт фиксированные 3000 мел-фреймов**. Решение:
   паддинг + обрезка выхода по `ceil(real_len/2)`.
5. **CTC требует T_out >= max(text_lens)**. С Whisper это 1500 фреймов
   для 30-секундного аудио — заведомо больше любого разумного текста.

---

## Дневник экспериментов (заполняется по мере)

### Эксперимент 1 — audio-only baseline
- **Цель**: установить точку отсчёта WER на чистом аудио.
- **Конфиг**: `mode=audio_only`, Whisper заморожен, нет видео-ветки.
- **Данные**: TBD
- **Результаты**: TBD

### Эксперимент 2 — full AVSR с cross-attention
- **Цель**: показать прирост над audio-only.
- **Конфиг**: `mode=av`, `fusion=cross_attention`.
- **Данные**: TBD
- **Результаты**: TBD

### Эксперимент 3 — аблация по типу слияния
- **Цель**: сравнить concat vs cross-attention.
- **Результаты**: TBD

### Эксперимент 4 — устойчивость к шуму
- **Цель**: ключевой результат курсовой.
- **Условия**: SNR = 0, 5, 10, 15 дБ.
- **Результаты**: TBD
