"""
Датасет AVSR + символьный токенизатор.

Один пример = одно высказывание. Манифест — jsonl (одна запись в строке):
  {"id":       "abc123",
   "audio":    "data/raw/.../abc123.wav",
   "video":    "data/raw/.../abc123.mp4",
   "lip_npy":  "data/processed/abc123_lips.npy",   # ОБЯЗАТЕЛЬНО предвычислен
   "text":     "hello world",
   "duration": 3.21}

Кропы губ должны быть закешированы в .npy (uint8, форма (T, 96, 96)).
Прогонять MediaPipe в __getitem__ — нельзя, слишком медленно.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import (
    LIP_SIZE,
    SAMPLE_RATE,
    waveform_to_mel,
)


log = logging.getLogger(__name__)


# =============================================================================
# CharTokenizer
# =============================================================================

class CharTokenizer:
    """
    Простой посимвольный токенизатор для английского.

    Алфавит фиксированный:
        index 0:     <blank>       — CTC blank
        index 1-26:  'a'..'z'      — латиница в нижнем регистре
        index 27:    ' '           — пробел
        index 28:    "'"           — апостроф (don't, it's, ...)

    Итого 29 токенов. По соглашению CTCLoss blank_id = 0.
    """

    BLANK_TOKEN: str = "<blank>"
    BLANK_ID: int = 0

    def __init__(self) -> None:
        # Алфавит. Порядок ВАЖЕН — id'шники прибиваются к индексам.
        self.chars: list[str] = [self.BLANK_TOKEN] + list("abcdefghijklmnopqrstuvwxyz '")
        self.char_to_id: dict[str, int] = {c: i for i, c in enumerate(self.chars)}
        self.id_to_char: dict[int, str] = {i: c for c, i in self.char_to_id.items()}
        # Регекс для нормализации
        self._allowed = re.compile(r"[^a-z' ]+")
        self._spaces = re.compile(r"\s+")

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def normalize(self, text: str) -> str:
        """
        Приводит текст к каноническому виду:
          - lowercase
          - все варианты апострофов/кавычек -> простой '
          - всё, что не a-z/пробел/апостроф, заменяется на пробел
          - повторные пробелы схлопываются
          - обрезаются пробелы по краям
        """
        text = text.lower()
        # Унификация апострофов разных видов
        for quote in ("‘", "’", "‛", "`", "ʼ"):
            text = text.replace(quote, "'")
        text = self._allowed.sub(" ", text)
        text = self._spaces.sub(" ", text).strip()
        return text

    def encode(self, text: str, normalize: bool = True) -> List[int]:
        """text -> [id, id, ...]. Blank в выход не попадает."""
        if normalize:
            text = self.normalize(text)
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids: Sequence[int], collapse_blanks: bool = False) -> str:
        """
        ids -> text. Игнорирует blank.
        collapse_blanks=False — просто отбрасываем blank (для уже декодированных
        CTC-выходов это правильно).
        collapse_blanks=True — также схлопываем подряд идущие одинаковые
        символы (полный CTC-декодинг raw-предсказаний).
        """
        out: list[str] = []
        prev: Optional[int] = None
        for i in ids:
            i = int(i)
            if i == self.BLANK_ID:
                prev = None
                continue
            if collapse_blanks and i == prev:
                continue
            ch = self.id_to_char.get(i)
            if ch is not None:
                out.append(ch)
            prev = i
        return "".join(out)


# =============================================================================
# AVSRDataset
# =============================================================================

class AVSRDataset(Dataset):
    """
    PyTorch-датасет AVSR.

    Args:
        manifest_path:  путь к jsonl-манифесту.
        tokenizer:      экземпляр CharTokenizer.
        max_duration:   фильтруем примеры длиннее этой длительности (сек).
        min_duration:   и короче этой (сек) — нужны, чтобы убрать мусор.
        load_video:     если False, видео не грузим (для audio-only бейзлайна).
        require_lip_cache: True (по умолчанию) — требуем закешированных .npy.
                        False — позволяем извлекать губы из видео налету
                        (медленно, только для отладки!).

    Возвращает в __getitem__:
        {
            "id":         str,
            "audio_mel":  Tensor (80, T_a),     float32
            "video":      Tensor (T_v, 1, H, W), float32 в [0, 1]
            "text_ids":   Tensor (L,),          int64
            "text":       str,
        }
    """

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: CharTokenizer,
        max_duration: float = 15.0,
        min_duration: float = 0.5,
        load_video: bool = True,
        require_lip_cache: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer
        self.load_video = load_video
        self.require_lip_cache = require_lip_cache

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Манифест не найден: {self.manifest_path}")

        # Загружаем и фильтруем
        all_entries: list[dict] = []
        with open(self.manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                all_entries.append(json.loads(line))

        filtered: list[dict] = []
        skipped_dur = 0
        for e in all_entries:
            dur = float(e.get("duration", 0.0))
            if dur and not (min_duration <= dur <= max_duration):
                skipped_dur += 1
                continue
            filtered.append(e)

        self.entries: list[dict] = filtered
        log.info(
            "Манифест %s: всего %d, после фильтра [%.1f, %.1f] сек — %d, "
            "отброшено по длительности %d",
            self.manifest_path.name, len(all_entries),
            min_duration, max_duration, len(filtered), skipped_dur,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _load_audio(self, path: str) -> torch.Tensor:
        """wav -> log-mel (80, T_a)."""
        # soundfile быстрее torchaudio.load для wav-файлов
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        wav_t = torch.from_numpy(np.ascontiguousarray(wav))
        return waveform_to_mel(wav_t, sample_rate=sr)

    def _load_video(self, entry: dict) -> torch.Tensor:
        """Возвращает (T_v, 1, 96, 96), float32 в [0, 1]."""
        if "lip_npy" in entry and Path(entry["lip_npy"]).exists():
            arr = np.load(entry["lip_npy"])
            # ожидаем (T, 96, 96) uint8 ИЛИ (T, 1, 96, 96) float
            if arr.ndim == 3:
                tensor = torch.from_numpy(arr).float() / 255.0
                tensor = tensor.unsqueeze(1)              # (T, 1, H, W)
            elif arr.ndim == 4:
                tensor = torch.from_numpy(arr).float()
                if tensor.max() > 1.5:                    # ещё uint8
                    tensor = tensor / 255.0
            else:
                raise ValueError(f"Странная форма {arr.shape} в {entry['lip_npy']}")
            return tensor
        if self.require_lip_cache:
            raise FileNotFoundError(
                f"Кеш губ не найден: {entry.get('lip_npy')}. "
                f"Прогоните scripts/prepare_data.py заранее, либо передайте "
                f"require_lip_cache=False (медленно)."
            )
        # Fallback: извлекаем губы из видео налету (только для отладки!)
        from src.data.preprocessing import video_to_lip_tensor
        return video_to_lip_tensor(entry["video"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        mel = self._load_audio(entry["audio"])
        text_ids = torch.tensor(self.tokenizer.encode(entry["text"]), dtype=torch.long)

        if self.load_video:
            video = self._load_video(entry)
        else:
            # zero-видео фиксированной длины — для audio-only режима
            video = torch.zeros(1, 1, LIP_SIZE, LIP_SIZE, dtype=torch.float32)

        return {
            "id": str(entry.get("id", idx)),
            "audio_mel": mel,
            "video": video,
            "text_ids": text_ids,
            "text": entry["text"],
        }
