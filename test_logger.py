"""
Test Script for Logger Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

from src.logger import get_logger
from src.config import LOG_FILE

logger = get_logger("LoggerTest")


def main():

    print("=" * 60)
    print("LOGGER TEST")
    print("=" * 60)

    logger.info("Logger initialized successfully.")

    logger.warning("This is a warning message.")

    logger.error("This is an error message.")

    logger.debug(
        "This debug message should not appear "
        "unless log level is DEBUG."
    )

    print(f"\nLog file location:\n{LOG_FILE}")

    print("\nRegistered Handlers:")

    for handler in logger.handlers:
        print(type(handler).__name__)

    print("\nLogger Test Passed Successfully")

    print("=" * 60)


if __name__ == "__main__":

    main()