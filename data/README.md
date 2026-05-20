# Данные: MUAVIC (английский подсет)

Используем датасет **MUAVIC** (Multilingual Audio-Visual Corpus) от Meta:
аудио-видео речь с TED-докладов, открытый доступ.

Репозиторий: https://github.com/facebookresearch/muavic

## Что качаем

Для курсовой берём только **английский** + ограниченный объём (~10 ч train,
~1 ч val). Большего бесплатный Colab всё равно не потянет.

## Шаги (исполнять на машине с интернетом — Colab/локально)

### 1. Клонируем репозиторий MUAVIC

```bash
git clone https://github.com/facebookresearch/muavic.git
cd muavic
pip install -r requirements.txt
```

### 2. Скачиваем подсет

MUAVIC построен поверх mTEDx (аудио + транскрипции). Видео он подтягивает
с YouTube/TED по их инструкции. Полный английский — ~430 часов, нам столько
не нужно.

Сокращаем вручную, ограничив длительность train ~10 часов:

```bash
# (в репозитории muavic)
python prepare/get_video.py --languages en --output_dir /content/muavic_raw
# дождаться скачивания, потом обрезать train-список:
head -n 4000 /content/muavic_raw/en/train.tsv > /content/muavic_raw/en/train_small.tsv
```

Точные команды могут меняться — сверяйтесь со свежим README в muavic.

### 3. Складываем в наш проект

Положить в `data/raw/muavic_en/` структуру:

```
data/raw/muavic_en/
├── train/
│   ├── audio/*.wav      # 16 кГц, моно
│   ├── video/*.mp4      # 25 FPS
│   └── train.tsv         # id, duration, transcript
├── val/
│   └── ...
└── test/
    └── ...
```

### 4. Предобработка (запускается один раз)

```bash
python scripts/prepare_data.py \
    --src data/raw/muavic_en/train \
    --dst data/processed/train \
    --manifest-out data/manifests/train.jsonl \
    --split train
```

Скрипт:
1. Для каждого видео вытаскивает губы MediaPipe'ом, сохраняет в `.npy`.
2. Собирает строки манифеста с путями и транскрипцией.

## Альтернатива (если MUAVIC не зайдёт)

- **LRS3**: эталонный бенчмарк, но нужно подать заявку в Oxford VGG
  (https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html). Могут
  отказать или тянуть неделями — для месячного дедлайна рискованно.
- **GRID**: маленький (1000 предложений, фиксированная грамматика), но
  открытый и быстрый. Хорош для отладки пайплайна, но для серьёзных
  выводов в курсовой слабоват.

## Sanity check для одного примера

После скачивания убедимся, что хотя бы один пример читается:

```python
import soundfile as sf, cv2
wav, sr = sf.read("data/raw/muavic_en/train/audio/EXAMPLE.wav")
print(f"audio: {len(wav)/sr:.2f} sec, sr={sr}")
cap = cv2.VideoCapture("data/raw/muavic_en/train/video/EXAMPLE.mp4")
print(f"video frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}, "
      f"fps: {cap.get(cv2.CAP_PROP_FPS):.1f}")
```

Если оба прочитались — данные готовы к предобработке.
