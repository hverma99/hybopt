"""
Input checks for the ANN modules; they raise ``ValueError`` with a clear message.
(Checks on data and splits live in ``hybopt.data``.)
"""

import numpy as np


def positive_int(value, name: str) -> int:
    """
    Return ``value`` as an int if it is a positive integer (bool and float are rejected).
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)
