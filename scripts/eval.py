"""
Оценка обученной модели на валидации/тесте.

Запуск:
    python scripts/eval.py --checkpoint checkpoints/best.pt \
                           --manifest data/manifests/val.jsonl \
                           --noise-snr 5    # дБ, для теста на шуме

(заглушка; реализация на дне 19)
"""
from __future__ import annotations
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--noise-snr", type=float, default=None,
                        help="Если задано — накладываем шум из MUSAN с таким SNR")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
