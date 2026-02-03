import logging
import os
import sys

from app.config import settings

_configured = False


def setup_logger(name: str) -> logging.Logger:
    """Setup root logger once with configured handlers."""
    global _configured

    if not _configured:
        root_logger = logging.getLogger()
        stream_level = logging._nameToLevel.get(settings.LOG_LEVEL.upper(), logging.INFO)
        file_level = stream_level
        if settings.LOG_FILE_PATH:
            file_level = logging._nameToLevel.get(
                settings.LOG_FILE_LEVEL.upper(), logging.DEBUG
            )
        root_level = min(stream_level, file_level)
        root_logger.setLevel(root_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(stream_level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

        if settings.LOG_FILE_PATH:
            os.makedirs(os.path.dirname(settings.LOG_FILE_PATH), exist_ok=True)
            file_mode = "w" if settings.LOG_FILE_OVERWRITE else "a"
            file_handler = logging.FileHandler(settings.LOG_FILE_PATH, mode=file_mode)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        _configured = True

    return logging.getLogger(name)
