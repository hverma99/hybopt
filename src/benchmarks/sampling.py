"""
Sample the benchmark functions into labeled data sets, and read/write those samples as CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from benchmarks.functions import TestFunction
from hybopt.data import Dataset


def sample_function(
    function: TestFunction, n_points: int, seed: int, design: str = "lhs"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample a test function at n_points.

    Inputs
    ------
    function    : Function to sample, e.g. ``get_function("peaks")``.
    n_points    : Number of samples.
    seed        : Random seed.
    design      : "lhs" for Latin Hypercube Sampling or "uniform" for uniform random sampling.

    Returns
    -------
    x : Inputs of shape (n_points, dim), inside [function.lower_bound, function.upper_bound].
    y : Function values of shape (n_points, ).
    """
    rng = np.random.default_rng(seed)
    dim = len(function.lower_bound)
    if design == "lhs":
        unit = qmc.LatinHypercube(d=dim, rng=rng).random(n_points)
    elif design == "uniform":
        unit = rng.random((n_points, dim))
    else:
        raise ValueError(
            f"unknown sampling method {design!r}; only 'lhs' and 'uniform' supported"
        )
    x = qmc.scale(unit, function.lower_bound, function.upper_bound)
    return x, function(x)


def sample_dataset(
    function: TestFunction, n_points: int, seed: int, design: str = "lhs"
) -> Dataset:
    """
    Sample a test function into a ``Dataset`` named after it, with its domain as the input box.
    """
    x, y = sample_function(function, n_points, seed, design)
    return Dataset(x, y, function.lower_bound, function.upper_bound, name=function.name)


def save_samples(x: np.ndarray, y: np.ndarray, path: Path) -> None:
    """
    Save samples (CSV) with columns x1, ..., xn, then y (one output) or y1, ..., yk (several).
    Floats are written with 17 significant digits, so ``load_samples`` returns them exactly.
    """
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    y = np.asarray(y, dtype=float).reshape(len(y), -1)
    y_columns = ["y"] if y.shape[1] == 1 else [f"y{i + 1}" for i in range(y.shape[1])]
    columns = [f"x{i + 1}" for i in range(x.shape[1])] + y_columns

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(np.hstack([x, y]), columns=columns).to_csv(
        path, index=False, float_format="%.17g"
    )


def load_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load samples saved by ``save_samples``.
    Returns x of shape (n, dim) and y of shape (n,) for one output or (n, k) for several.
    """
    data = pd.read_csv(path, float_precision="round_trip")
    for column in data.columns:
        if column[:1] not in ("x", "y"):
            raise ValueError(
                f"{path} has an unexpected column {column!r}; expected x1, ..., xn and y or "
                "y1, ..., yk (files written by older versions must be regenerated)"
            )
    x = data.filter(regex="^x").to_numpy(dtype=float)
    y = data.filter(regex="^y").to_numpy(dtype=float)
    return x, y[:, 0] if y.shape[1] == 1 else y
