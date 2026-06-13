"""
Быстрый audio-only препроцессинг GRID прямо в сессию (без MediaPipe-губ).

Зачем: для audio-only эксперимента кропы губ не нужны, а извлечение губ
(MediaPipe) — самая долгая часть prepare_grid.py (часы). Здесь только:
качаем видео-zip с Sheffield → ffmpeg извлекает 16 кГц моно wav → манифесты.

Текст берётся из имени файла по фиксированной грамматике GRID (command-color-
preposition-letter-digit-adverb), длительность — из размера wav-файла
(PCM16 mono 16k: (size-44)/32000). Манифест GRID-совместим с AVSRDataset
в режиме audio_only (поля video/lip_npy не требуются).

Запуск:
    python scripts/prepare_grid_audio.py --base /content/grid --val-speakers s2
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

SHEFFIELD = "https://spandh.dcs.shef.ac.uk/gridcorpus"

# Грамматика GRID для декодирования имени файла в текст
CMD = {"b": "bin", "l": "lay", "p": "place", "s": "set"}
COL = {"b": "blue", "g": "green", "r": "red", "w": "white"}
PREP = {"a": "at", "b": "by", "i": "in", "w": "with"}
DIG = {"z": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
       "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
       "0": "zero"}
ADV = {"a": "again", "n": "now", "p": "please", "s": "soon"}


def stem_to_text(stem: str) -> str:
    """'bbaf2n' → 'bin blue at f two now'."""
    c = stem
    return " ".join([CMD[c[0]], COL[c[1]], PREP[c[2]], c[3], DIG[c[4]], ADV[c[5]]])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("/content/grid"),
                   help="корневая папка для raw/, processed/, manifests/")
    p.add_argument("--speakers", default="all10",
                   help="'all10' (s1..s10) или список через запятую: 's1,s2'")
    p.add_argument("--val-speakers", default="s2",
                   help="спикеры в val (speaker-disjoint), через запятую")
    p.add_argument("--min-dur", type=float, default=0.1)
    p.add_argument("--max-dur", type=float, default=8.0)
    args = p.parse_args()

    if args.speakers == "all10":
        speakers = [f"s{i}" for i in range(1, 11)]
    else:
        speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    val = {s.strip() for s in args.val_speakers.split(",") if s.strip()}

    recs = []
    for spk in speakers:
        raw = args.base / "raw" / spk
        raw.mkdir(parents=True, exist_ok=True)
        if not any(raw.rglob("*.mpg")):
            z = raw / "v.zip"
            subprocess.run(["wget", "-q", "-O", str(z),
                            f"{SHEFFIELD}/{spk}/video/{spk}.mpg_vcd.zip"])
            subprocess.run(["unzip", "-qo", str(z), "-d", str(raw)])
            z.unlink(missing_ok=True)

        wd = args.base / "processed" / spk / "wav"
        wd.mkdir(parents=True, exist_ok=True)
        for mpg in sorted(raw.rglob("*.mpg")):
            wav = wd / (mpg.stem + ".wav")
            if not wav.exists():
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                                "-i", str(mpg), "-ar", "16000", "-ac", "1",
                                str(wav)])

        n = 0
        for e in os.scandir(wd):
            if not e.name.endswith(".wav"):
                continue
            stem = e.name[:-4]
            try:
                text = stem_to_text(stem)
            except (KeyError, IndexError):
                continue
            dur = round((e.stat().st_size - 44) / 32000.0, 3)
            if not (args.min_dur <= dur <= args.max_dur):
                continue
            recs.append({
                "id": f"{spk}_{stem}", "speaker": spk,
                "audio": str(wd / e.name), "text": text, "duration": dur,
            })
            n += 1
        print(f"{spk}: {n} wav+manifest", flush=True)

    train = [r for r in recs if r["speaker"] not in val]
    val_recs = [r for r in recs if r["speaker"] in val]
    md = args.base / "manifests"
    md.mkdir(parents=True, exist_ok=True)
    for name, rs in [("train.jsonl", train), ("val.jsonl", val_recs)]:
        with open(md / name, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ train={len(train)} val={len(val_recs)} → {md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
