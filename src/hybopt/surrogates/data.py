import csv
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc
from torch.utils.data import DataLoader, TensorDataset, random_split

from hybopt.surrogates.functions import TestFunction


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


def save_samples(x: np.ndarray, y: np.ndarray, path: Path) -> None:
    """
    Save samples (CSV) with columns x1, ..., xn, y
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"x{i + 1}" for i in range(x.shape[1])] + ["y"])
        for xi, yi in zip(x, y):
            writer.writerow([f"{v:.17g}" for v in xi] + [f"{yi:.17g}"])


def load_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load samples saved by ``save_samples``
    Returns x of shape (n, dim) and y of shape (n,).
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[:, :-1], data[:, -1]


class DataModule:
    """
    Split labeled samples into train/val/test and build their DataLoaders.

    Inputs are scaled from the box domain [lower_bound, upper_bound] to [-1, 1], and outputs are
    standardized with the mean and std of the training split.

    The constants are kept in ``normalization`` so a trained network can be mapped back to raw units.

    Inputs
    ------
    x : np.ndarray
        Raw inputs of shape (n, dim).
    y : np.ndarray
        Raw outputs of shape (n,).
    lower_bound, upper_bound : sequence of float
        Box domain of the inputs.
    batch_size : int
        Number of samples per batch.
    split : dict
        Fractions {"train": ..., "val": ..., "test": ...} that sum to 1.
    random_state : int, optional, default=0
        Seed for the split; the same seed always gives the same split.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lower_bound,
        upper_bound,
        batch_size: int,
        split: dict,
        random_state: int = 0,
    ):
        self.x = x
        self.y = y
        self.lower_bound = np.asarray(lower_bound, dtype=float)
        self.upper_bound = np.asarray(upper_bound, dtype=float)
        self.batch_size = batch_size

        subsets = random_split(
            TensorDataset(torch.arange(len(y))),
            [split["train"], split["val"], split["test"]],
            generator=torch.Generator().manual_seed(random_state),
        )

        self.idx = {
            name: np.array(subset.indices)
            for name, subset in zip(("train", "val", "test"), subsets)
        }

        y_train = y[self.idx["train"]]
        self.normalization = {
            "x_shift": ((self.upper_bound + self.lower_bound) / 2).tolist(),
            "x_scale": ((self.upper_bound - self.lower_bound) / 2).tolist(),
            "y_shift": [float(y_train.mean())],
            "y_scale": [float(y_train.std())],
        }

    def labeled_data_loader(self, shuffle_seed: int = 0) -> tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        """
        Construct DataLoaders for the training, validation and test splits.

        Inputs
        ------
        shuffle_seed : Seed for shuffling the training batches.

        Returns
        -------
            Batches of normalized (X, Y) with shapes (batch_size, dim) and (batch_size, 1).
        """
        norm = {k: np.array(v) for k, v in self.normalization.items()}

        def make_loader(name: str, shuffle: bool):
            x = self.x[self.idx[name]]
            y = self.y[self.idx[name]]
            X_t = torch.tensor(
                (x - norm["x_shift"]) / norm["x_scale"], dtype=torch.float32
            )
            Y_t = torch.tensor(
                (y - norm["y_shift"]) / norm["y_scale"], dtype=torch.float32
            )
            return DataLoader(
                dataset=TensorDataset(X_t, Y_t.unsqueeze(1)),
                batch_size=self.batch_size,
                shuffle=shuffle,
                generator=(
                    torch.Generator().manual_seed(shuffle_seed) if shuffle else None
                ),
            )

        train_loader = make_loader("train", shuffle=True)
        val_loader = make_loader("val", shuffle=False)
        test_loader = make_loader("test", shuffle=False)

        return train_loader, val_loader, test_loader
