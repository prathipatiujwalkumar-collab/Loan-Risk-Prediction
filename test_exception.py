import sys

from src.exceptions import LoanRiskException


def divide_numbers():
    try:
        result = 10 / 0
        return result
    except Exception as error:
        raise LoanRiskException(error, sys)


divide_numbers()