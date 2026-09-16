"""
Test script for LoanRiskException

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

import sys

from src.exceptions import LoanRiskException


def divide_numbers():

    try:

        result = 10 / 0

        return result

    except Exception as error:

        raise LoanRiskException(error, sys)


def main():

    try:

        divide_numbers()

    except LoanRiskException as error:

        print(error)

        print("\nException Test Successful")


if __name__ == "__main__":

    main()