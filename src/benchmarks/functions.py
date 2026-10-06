"""
Analytical benchmark test functions.

Every function takes an array of shape (n_samples, dim) and returns shape (n_samples,).
To add a function, define it below and add a ``TestFunction`` entry to ``FUNCTIONS``.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TestFunction:

    __test__ = False

    name: str
    # Numerical objective value at each sample; input shape is (n_samples, dim).
    function: Callable[[np.ndarray], np.ndarray]

    lower_bound: tuple[float, ...]  # box domain, lower bounds
    upper_bound: tuple[float, ...]  # box domain, upper bounds
    x_min: tuple[tuple[float, ...], ...]  # known global minimizers
    function_min: float  # known global minimum value

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Validate the input shape and evaluate the objective on batched samples.
        """
        # Convert scalars or 1D arrays into a 2D batch with one row per sample.
        x = np.atleast_2d(np.asarray(x, dtype=float))
        dim = len(self.lower_bound)
        if x.shape[1] != dim:
            raise ValueError(f"{self.name} expects {dim} inputs, got {x.shape[1]}")
        return self.function(x)


def _peaks(x: np.ndarray) -> np.ndarray:
    """
    Multi-modal 2D landscape with a sharp global peak and several local maxima.
    """
    x1, x2 = x[:, 0], x[:, 1]
    return (
        3 * (1 - x1) ** 2 * np.exp(-(x1**2) - (x2 + 1) ** 2)
        - 10 * (x1 / 5 - x1**3 - x2**5) * np.exp(-(x1**2) - x2**2)
        - np.exp(-((x1 + 1) ** 2) - x2**2) / 3
    )


def _ackley(x: np.ndarray) -> np.ndarray:
    """
    Smooth basin-shaped function with many local minima near the center.
    """
    x1, x2 = x[:, 0], x[:, 1]
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
        + np.e
        + 20
    )


def _himmelblau(x: np.ndarray) -> np.ndarray:
    """
    Four-well 2D function with several distinct global minima.
    """
    x1, x2 = x[:, 0], x[:, 1]
    return (x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2


# Registry of available benchmark functions used throughout the project.
FUNCTIONS: dict[str, TestFunction] = {
    function.name: function
    for function in [
        TestFunction(
            "peaks",
            _peaks,
            (-2.0, -2.0),
            (2.0, 2.0),
            ((0.228, -1.626),),
            -6.551,
        ),
        TestFunction(
            "ackley",
            _ackley,
            (-3.5, -3.5),
            (3.5, 3.5),
            ((0.0, 0.0),),
            0.0,
        ),
        TestFunction(
            "himmelblau",
            _himmelblau,
            (-5.0, -5.0),
            (5.0, 5.0),
            ((3.0, 2.0), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)),
            0.0,
        ),
    ]
}


def get_function(name: str) -> TestFunction:
    """Return a registered function by name, raising a clear error when missing."""
    try:
        return FUNCTIONS[name]
    except KeyError:
        raise KeyError(
            f"unknown function {name!r}; available: {sorted(FUNCTIONS)}"
        ) from None
