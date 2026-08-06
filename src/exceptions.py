"""
Custom exception class for the Loan Risk Prediction project.

This module provides a standardized exception that captures
useful debugging information such as the filename and line number
where an error occurred.
"""

# --------------------------------------------------
# Standard Library Imports
# --------------------------------------------------

import sys


class LoanRiskException(Exception):
    """
    Custom exception class for the project.

    This exception wraps the original exception and provides
    additional debugging information.
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = self._get_detailed_error_message(
            error_message,
            error_detail,
        )

    @staticmethod
    def _get_detailed_error_message(
        error: Exception,
        error_detail: sys,
    ) -> str:
        """
        Generate a detailed error message.

        Parameters
        ----------
        error : Exception
            Original exception.

        error_detail : sys
            System module used to access exception information.

        Returns
        -------
        str
            Detailed error message.
        """

        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is None:
            return str(error)

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            "\n"
            "========== Loan Risk Exception ==========\n"
            f"File        : {file_name}\n"
            f"Line        : {line_number}\n"
            f"Error Type  : {type(error).__name__}\n"
            f"Message     : {error}\n"
            "========================================="
        )

    def __str__(self) -> str:
        return self.error_message