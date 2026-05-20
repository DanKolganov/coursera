"""
Загрузка YAML-конфигов через OmegaConf.

Зачем: все гиперпараметры (lr, batch_size, размеры слоёв) держим в YAML, а не
в коде. Так удобно запускать разные эксперименты без правки исходников.

Использование:
    from src.utils.config import load_config
    cfg = load_config("configs/avsr_baseline.yaml")
    print(cfg.training.lr)
"""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(path: str | Path) -> DictConfig:
    """Загружает YAML и возвращает иерархический конфиг."""
    cfg = OmegaConf.load(path)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Сохраняет конфиг в YAML (полезно вместе с чекпоинтом)."""
    with open(path, "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)
