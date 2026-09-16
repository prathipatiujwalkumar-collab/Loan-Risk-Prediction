"""
Production-grade Exceptions Module

Author: Ujwal
Project: Intelligent Loan Risk Assessment Platform
"""

# ==================================================
# Standard Library Imports
# ==================================================

import sys
import traceback


class LoanRiskException(Exception):
    """
    Custom exception class for the Loan Risk Prediction project.

    Wraps the original exception and provides detailed debugging
    information including filename, function name, line number,
    exception type, and error message.
    """

    def __init__(
        self,
        error: Exception,
        error_detail=sys,
    ) -> None:

        super().__init__(str(error))

        self.error_message = self._build_error_message(
            error,
            error_detail,
        )

    @staticmethod
    def _build_error_message(
        error: Exception,
        error_detail=sys,
    ) -> str:
        """
        Construct a detailed exception message.
        """

        _, _, tb = error_detail.exc_info()

        if tb is None:
            return str(error)

        last_tb = traceback.extract_tb(tb)[-1]

        return (
            "\n"
            "=====================================================\n"
            "Loan Risk Prediction Exception\n"
            "=====================================================\n"
            f"File      : {last_tb.filename}\n"
            f"Function  : {last_tb.name}\n"
            f"Line      : {last_tb.lineno}\n"
            f"Type      : {type(error).__name__}\n"
            f"Message   : {error}\n"
            "====================================================="
        )

    def __str__(self) -> str:

        return self.error_message

    def __repr__(self) -> str:

        return self.error_message