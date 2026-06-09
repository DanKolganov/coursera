# Финальная инструкция: GRID → обучение → результаты

Открой этот файл у себя в Cursor/VS Code на маке. Ниже — последовательность,
в которой работать. Никаких других ноутбуков и инструкций больше не нужно.

---

## Шаг 0. На маке: закоммитить ВСЕ мои правки

В Терминале, в папке проекта:

```bash
cd /Users/d3zzle/Desktop/cursach/cursera_claude

# Удалить мусор (sandbox не дал мне это сделать)
rm -f avsr_src.zip PROGRESS.md zi5zfJph 2>/dev/null

# Очистка устаревших ноутбуков (если ещё лежат)
rm -f notebooks/01_preprocessing_demo.md \
      notebooks/02_train_in_colab.md \
      notebooks/03_one_paste_train.md \
      notebooks/04_grid_av_train.md \
      notebooks/train_colab.ipynb \
      notebooks/muavic_colab.ipynb 2>/dev/null

# Запушить ВСЁ что менялось
git add -A
git commit -m "consolidation: GRID pipeline + fixes (sys.path, recursive .mpg/.align)"
git push
```

**Проверь, что push прошёл** — `git log --oneline -3` должно показать твой свежий коммит.

---

## Шаг 1. В Colab: подтянуть свежий код

В уже открытой сессии — одна ячейка:

```python
%cd /content/coursera
!git pull

# Проверка что фикс подтянулся
!grep "не найдено .mpg" scripts/prepare_grid.py | head -1
```

Должна вывестись строка `log.warning("[%s] не найдено .mpg в %s — пропуск", ...)`.
Если grep пуст — `git pull` не подтянул, разбираемся отдельно.

---

## Шаг 2. Препроцессинг GRID (раз архивы уже на Drive)

```python
# Архивы уже скачаны — пропускаем download, только MediaPipe
!cd /content/coursera && PYTHONPATH=/content/coursera python scripts/prepare_grid.py \
    --speakers smoke \
    --base /content/drive/MyDrive/avsr/grid \
    --skip-download
```

**Время**: ~40 минут на одного спикера (MediaPipe на CPU, 1000 видео × ~75 кадров).

**Что должны увидеть в логах:**
```
INFO | [s1] найдено: 1000 видео, 1000 алайнов
s1 preprocess: 100%|████████| 1000/1000 [40:00<00:00, ...]
INFO | [s1] записей: 998
INFO | Всего обработано: 998 записей
INFO | Разбиение: random 90%/10%
INFO | Манифесты: train=898 (~45 мин), val=100 (~5 мин)
```

Если опять `WARNING нет видео — пропуск` или `найдено: 0 видео` —
прислать вывод этой ячейки:

```python
import subprocess
print(subprocess.check_output([
    "find", "/content/drive/MyDrive/avsr/grid/raw/s1",
    "-maxdepth", "4", "-type", "f"
]).decode()[:2500])
```

---

## Шаг 3. Эксперимент 1: audio-only baseline

```python
# Готовим конфиг под наш Drive
import yaml
with open("configs/grid_audio_only.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["train_manifest"] = "/content/drive/MyDrive/avsr/grid/manifests/train.jsonl"
cfg["data"]["val_manifest"]   = "/content/drive/MyDrive/avsr/grid/manifests/val.jsonl"
cfg["data"]["min_duration"]   = 0.1  # GRID-видео ~3 сек
cfg["experiment"]["output_dir"] = "/content/drive/MyDrive/avsr/checkpoints/grid_audio_only"
with open("configs/grid_audio_only_run.yaml", "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)

# Запуск обучения
!cd /content/coursera && PYTHONPATH=/content/coursera python scripts/train.py \
    --config configs/grid_audio_only_run.yaml
```

**Время**: ~15-20 минут на T4 (5 эпох, ~900 примеров).

**Что должны увидеть:**
```
INFO | Vocab size: 29
INFO | Датасеты: train=898, val=100 примеров
INFO | Параметров всего: 88M (обучаемых: 30M, заморожено: 58M)
Epoch 000 [train]: loss=2.31, lr=1e-4
Epoch 001 [train]: loss=1.45, lr=1e-4
...
Epoch 004  val_loss=0.32  WER=0.045  CER=0.018
  ✓ Новый лучший WER=0.045, чекпоинт сохранён
```

**Скопируй мне финальные числа** (последняя строка с `WER=`).

---

## Шаг 4. Эксперимент 2: AV cross-attention

```python
import yaml
with open("configs/grid_av.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["train_manifest"] = "/content/drive/MyDrive/avsr/grid/manifests/train.jsonl"
cfg["data"]["val_manifest"]   = "/content/drive/MyDrive/avsr/grid/manifests/val.jsonl"
cfg["data"]["min_duration"]   = 0.1
cfg["experiment"]["output_dir"] = "/content/drive/MyDrive/avsr/checkpoints/grid_av"
with open("configs/grid_av_run.yaml", "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)

!cd /content/coursera && PYTHONPATH=/content/coursera python scripts/train.py \
    --config configs/grid_av_run.yaml
```

**Время**: ~30-45 минут на T4 (8 эпох с видео-веткой).

Финальные числа — снова сюда.

---

## Шаг 5. Eval с шумом (главный результат курсовой)

```python
import subprocess, os, json

AUDIO_ONLY = "/content/drive/MyDrive/avsr/checkpoints/grid_audio_only/best.pt"
AV_CKPT    = "/content/drive/MyDrive/avsr/checkpoints/grid_av/best.pt"
VAL_MANI   = "/content/drive/MyDrive/avsr/grid/manifests/val.jsonl"
RESULTS    = "/content/drive/MyDrive/avsr/grid/results"
os.makedirs(RESULTS, exist_ok=True)

env = {**os.environ, "PYTHONPATH": "/content/coursera"}

results = {}
for ckpt, name in [(AUDIO_ONLY, "audio_only"), (AV_CKPT, "av_crossattn")]:
    results[name] = {}
    for snr in [None, 20, 15, 10, 5, 0]:
        snr_str = "clean" if snr is None else f"snr{snr}"
        out_json = f"{RESULTS}/{name}_{snr_str}.json"
        cmd = ["python", "scripts/eval.py",
               "--checkpoint", ckpt,
               "--manifest", VAL_MANI,
               "--output-json", out_json]
        if snr is not None:
            cmd += ["--noise-snr", str(snr)]
        print(f"\n=== {name} | {snr_str} ===")
        subprocess.run(cmd, env=env, cwd="/content/coursera")
        if os.path.exists(out_json):
            with open(out_json) as f:
                r = json.load(f)
            results[name][snr_str] = {"wer": r["wer"], "cer": r["cer"]}

with open(f"{RESULTS}/summary.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n=== ИТОГ ===")
print(json.dumps(results, indent=2))
```

**Время**: ~10 минут. Скопируй сюда финальный JSON-блок.

---

## Что я делаю когда получу числа

1. Заполняю таблицы 4.3, 4.4, 4.5, 4.6 в `docs/coursework/04_эксперименты.md`.
2. Подставляю реальные значения в заключение `05_заключение.md` (вместо `N процентных пунктов`, `X.XX`, `YYY мс`).
3. Пересобираю `курсовая_работа.docx` через `scripts/build_coursework.py`.
4. Делаю запись в `docs/REPORT.md`.

Тебе останется только открыть `курсовая_работа.docx`, заменить заглушки
титульника (`[Название университета]` и т.д.), при желании прогнать
автогенерируемое оглавление через Word.

---

## Если что-то падает

| Что вижу | Что делать |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Проверь, что в команде есть `PYTHONPATH=/content/coursera` |
| `Манифест не найден` | `prepare_grid.py` ещё не отработал успешно |
| `train=0 примеров` | манифест пустой — снова `prepare_grid.py` или диагностика find |
| `CUDA out of memory` | в конфиге `batch_size: 2` и `grad_accum: 8` |
| Colab отключился | новая сессия → шаги 1 → 3 с флагом `--resume /content/drive/.../last.pt` |
| MediaPipe не находит лицо в >50% кадров | дай знать, посмотрим |

---

## Список файлов в репо, которые тебе НУЖНЫ

В Colab будут подтягиваться эти файлы (всё через `git pull`):

- `scripts/prepare_grid.py` — скачивание + препроцессинг GRID
- `scripts/train.py` — точка входа обучения
- `scripts/eval.py` — оценка с опциональным шумом
- `configs/grid_audio_only.yaml` — шаблон для Эксп. 1
- `configs/grid_av.yaml` — шаблон для Эксп. 2
- `src/...` — весь код модели (Whisper-encoder, видео-энкодер, fusion, CTC)
- `INSTRUCTION.md` — этот файл (можешь читать его и в Colab через `!cat`)

Какие файлы **НЕ нужны для прогона** (но смотри если хочется деталей):

- `notebooks/grid_colab.md` — расширенная версия этой инструкции
- `docs/REPORT.md` — рабочий журнал
- `docs/coursework/*.md` + `курсовая_работа.docx` — главы курсовой

---

## Сводно — что ты делаешь по времени

1. **Сейчас (5 мин):** Шаг 0 — push с мака.
2. **Сейчас (1 мин):** Шаг 1 — git pull в Colab.
3. **~40 мин:** Шаг 2 — препроцессинг (можешь не сидеть рядом).
4. **~20 мин:** Шаг 3 — audio-only.
5. **~40 мин:** Шаг 4 — AV.
6. **~10 мин:** Шаг 5 — eval с шумом.

**Итого ~2 часа** до полного набора чисел для курсовой.

После — присылаешь логи, я подставляю в `.docx`, ты сдаёшь.
