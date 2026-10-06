"""
Seeded train/val/test splits.
"""

import math
import warnings

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

SPLITS = ("train", "val", "test")


def split_fractions(split) -> dict:
    """
    Return train/val/test fractions as a dict.
    Every fraction must be between 0 and 1 (exclusive) and together they must sum to 1.
    """
    if isinstance(split, dict):
        if set(split) != set(SPLITS):
            raise ValueError(
                f"split must have exactly the keys {list(SPLITS)}, got {sorted(split)}"
            )
        fractions = {name: split[name] for name in SPLITS}
    elif isinstance(split, (list, tuple)) and len(split) == 3:
        fractions = dict(zip(SPLITS, split))
    else:
        raise ValueError(
            "split must be a dict {'train', 'val', 'test'} or a (train, val, test) "
            f"sequence of fractions, got {split!r}"
        )

    for name, fraction in fractions.items():
        is_number = isinstance(fraction, (int, float, np.number)) and not isinstance(
            fraction, bool
        )
        if not is_number or not 0 < fraction < 1:
            raise ValueError(
                f"split fraction for {name!r} must be a number between 0 and 1, got {fraction!r}"
            )

    total = sum(fractions.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(
            f"split fractions must sum to 1, got {total:g} from {fractions}"
        )

    return {name: float(fraction) for name, fraction in fractions.items()}


def split_indices(n_samples: int, split, seed: int = 0) -> dict[str, np.ndarray]:
    """
    Randomly assign the rows 0, ..., n_samples - 1 to train/val/test.

    Inputs
    ------
    n_samples   : Number of rows.
    split       : Fractions {"train", "val", "test"} or (train, val, test), summing to 1.
    seed        : Random seed; the same seed always gives the same split.

    Returns
    -------
    {"train": idx, "val": idx, "test": idx}
    """
    fractions = split_fractions(split)
    with (
        warnings.catch_warnings()
    ):  # an empty split gets the clearer error below instead
        warnings.filterwarnings(
            "ignore", message="Length of split", category=UserWarning
        )
        subsets = random_split(
            TensorDataset(torch.arange(n_samples)),
            [fractions[name] for name in SPLITS],
            generator=torch.Generator().manual_seed(seed),
        )

    idx = {
        name: np.array(subset.indices, dtype=int)
        for name, subset in zip(SPLITS, subsets)
    }
    for name, rows in idx.items():
        if len(rows) < 2:
            raise ValueError(
                f"the {name} split gets {len(rows)} of {n_samples} samples; it needs "
                "at least 2, so use more data or a larger fraction"
            )
    return idx
