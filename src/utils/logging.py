"""
Единый логгер для всего проекта.

Зачем: чтобы во всех модулях писать `log.info(...)` одинаково, и потом одной
строкой переключать вывод в файл или менять формат.
"""
import logging
import sys


def get_logger(name: str = "avsr", level: int = logging.INFO) -> logging.Logger:
    """Возвращает настроенный логгер. Повторные вызовы возвращают тот же объект."""
    logger = logging.getLogger(name)
    if logger.handlers:  # уже настроен — не дублируем хендлеры
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger
