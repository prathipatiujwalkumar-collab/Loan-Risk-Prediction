"""
Production-grade Logger Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Standard Library Imports
# ==================================================

import logging
from logging.handlers import RotatingFileHandler

# ==================================================
# Local Imports
# ==================================================

from src.config import LOG_DIR

# ==================================================
# Log Configuration
# ==================================================

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = LOG_DIR / "loan_risk_prediction.log"

DEFAULT_LOG_LEVEL = logging.INFO


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(DEFAULT_LOG_LEVEL)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(filename)s:%(lineno)d | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ==================================================
    # Console Handler
    # ==================================================

    console_handler = logging.StreamHandler()

    console_handler.setLevel(DEFAULT_LOG_LEVEL)

    console_handler.setFormatter(formatter)

    # ==================================================
    # Rotating File Handler
    # ==================================================

    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=5 * 1024 * 1024,      # 5 MB
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setLevel(DEFAULT_LOG_LEVEL)

    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    logger.propagate = False

    return logger