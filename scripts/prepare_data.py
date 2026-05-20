"""
Один раз прогнать на сыром датасете:
  1) Для каждого видео найти губы через MediaPipe, сохранить .npy кроп.
  2) Собрать манифест train/val.jsonl (audio, video, lip_npy, text, duration).

Это нужно сделать ОДИН РАЗ, потом обучение не трогает MediaPipe.

(заглушка; реализация на дне 4-5)
"""
from __future__ import annotations
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="папка с сырым датасетом")
    parser.add_argument("--dst", required=True, help="куда сохранять .npy")
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
