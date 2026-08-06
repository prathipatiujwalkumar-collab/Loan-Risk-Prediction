"""
Logging configuration for the Loan Risk Prediction project.

This module provides a reusable project-wide logger that writes logs
to both the console and a log file.
"""

import logging
from pathlib import Path

from src.config import LOG_DIR

# ------------------------------------------------------------------
# Create logs directory if it doesn't exist
# ------------------------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "loan_risk_prediction.log"


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Name of the logger (usually __name__).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # -------------------------
    # Console Handler
    # -------------------------

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # -------------------------
    # File Handler
    # -------------------------

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger