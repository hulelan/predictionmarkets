import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler


def setup_logger(name: str = "prediction_markets", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Console handler (rich)
        console = RichHandler(rich_tracebacks=True, show_path=False)
        console.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console)

        # File handler — one log per run
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"run_{ts}.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(file_handler)

        logger.setLevel(getattr(logging, level.upper()))
    return logger


log = setup_logger()
