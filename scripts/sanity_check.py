"""
Проверка, что окружение собрано правильно.

Запуск:
    python scripts/sanity_check.py

Что проверяем:
  1) Все ключевые библиотеки импортируются.
  2) PyTorch видит GPU (или хотя бы корректно работает на CPU).
  3) MediaPipe и OpenCV запускаются (без реальных данных).
  4) Whisper можно загрузить с HuggingFace.

Если хоть что-то падает — фиксим до того, как идти дальше.
"""
from __future__ import annotations

import sys
import importlib
from typing import List, Tuple


REQUIRED = [
    # (имя модуля, дружелюбное имя)
    ("torch", "PyTorch"),
    ("torchaudio", "torchaudio"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers (HuggingFace)"),
    ("cv2", "opencv-python"),
    ("mediapipe", "mediapipe"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("jiwer", "jiwer"),
    ("omegaconf", "omegaconf"),
    ("numpy", "numpy"),
    ("einops", "einops"),
]


def check_imports() -> List[Tuple[str, str, bool, str]]:
    results = []
    for module, name in REQUIRED:
        try:
            m = importlib.import_module(module)
            version = getattr(m, "__version__", "?")
            results.append((module, name, True, version))
        except Exception as e:
            results.append((module, name, False, str(e)))
    return results


def check_gpu() -> None:
    import torch
    print(f"\n[torch] version: {torch.__version__}")
    print(f"[torch] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[torch] device: {torch.cuda.get_device_name(0)}")
        print(f"[torch] VRAM total: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("[torch] GPU не найдена. На CPU работать будет, но обучение —"
              " только на маленьком сабсете для отладки.")


def check_mediapipe() -> None:
    import mediapipe as mp
    import numpy as np
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = face_mesh.process(dummy)
    face_mesh.close()
    print(f"\n[mediapipe] FaceMesh запускается. На пустом кадре лиц: "
          f"{0 if not res.multi_face_landmarks else len(res.multi_face_landmarks)} (ожидается 0)")


def check_whisper() -> None:
    """Пробуем подгрузить только конфиг Whisper — это быстро и не качает веса."""
    from transformers import WhisperConfig
    cfg = WhisperConfig.from_pretrained("openai/whisper-small")
    print(f"\n[whisper] config small загружен: "
          f"d_model={cfg.d_model}, encoder_layers={cfg.encoder_layers}, "
          f"vocab_size={cfg.vocab_size}")


def main() -> int:
    print("=" * 60)
    print(" SANITY CHECK — окружение AVSR")
    print("=" * 60)

    results = check_imports()
    ok = True
    for module, name, success, info in results:
        if success:
            print(f"  [OK]   {name:30s} {info}")
        else:
            ok = False
            print(f"  [FAIL] {name:30s} {info}")

    if not ok:
        print("\nНе все библиотеки установлены. Запусти:")
        print("    pip install -r requirements.txt")
        return 1

    check_gpu()
    try:
        check_mediapipe()
    except Exception as e:
        print(f"[mediapipe] ОШИБКА: {e}")
    try:
        check_whisper()
    except Exception as e:
        print(f"[whisper] ОШИБКА (нужен интернет, если запускается впервые): {e}")

    print("\nВсё готово. Можно идти на день 3 (предобработка).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
