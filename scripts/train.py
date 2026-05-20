"""
Точка входа для обучения. Запуск:
    python scripts/train.py --config configs/avsr_baseline.yaml

(сейчас файл-заглушка; реализация на дне 13)
"""
from __future__ import annotations
import argparse
from pathlib import Path

# from src.utils.config import load_config
# from src.utils.logging import get_logger
# from src.data.dataset import AVSRDataset, CharTokenizer
# from src.data.collate import avsr_collate_fn
# from src.models.avsr_model import AVSRModel
# from src.training.loss import CTCLossWrapper
# from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    # TODO(day-13): загрузить cfg, собрать dataloaders, модель, trainer, .fit()
    raise NotImplementedError("Реализация на этапе обучения (день 13).")


if __name__ == "__main__":
    main()
