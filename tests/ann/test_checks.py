import numpy as np
import pytest

from hybopt.ann.checks import positive_int


# --- positive_int ---------------------------------------------------------------------


@pytest.mark.parametrize("value", [1, 2, 50, np.int64(3), np.int32(7)])
def test_positive_int_accepts_positive_integers(value):
    result = positive_int(value, "depth")
    assert result == value and type(result) is int


@pytest.mark.parametrize("value", [0, -1, 2.0, 2.5, "3", None, True, False, [2], np.float64(2)])
def test_positive_int_rejects_everything_else(value):
    with pytest.raises(ValueError, match="depth must be a positive integer"):
        positive_int(value, "depth")


def test_positive_int_names_the_argument_in_the_message():
    with pytest.raises(ValueError, match=r"width must be a positive integer, got -3"):
        positive_int(-3, "width")
