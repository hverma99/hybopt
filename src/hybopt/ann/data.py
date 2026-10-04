"""
ANN-specific data preparation: split a ``Dataset``, scale it, and batch it into DataLoaders.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hybopt.ann.checks import positive_int
from hybopt.data import Dataset, split_fractions, split_indices


class DataModule:
    """
    Split a data set into train/val/test and build their DataLoaders.

    Inputs are scaled from the data set's input box to [-1, 1], and every output is standardized
    with its mean and std on the training split. The constants are kept in ``normalization`` so a
    trained network can be mapped back to raw units.

    Inputs
    ------
    data          : The labeled ``Dataset``.
    batch_size    : Number of samples per batch.
    split         : Fractions {"train", "val", "test"} (or a 3-tuple) that sum to 1.
    random_state  : Seed for the split; the same seed always gives the same split.
    """

    def __init__(self, data: Dataset, batch_size: int, split, random_state: int = 0):
        if not isinstance(data, Dataset):
            raise ValueError(
                f"data must be a hybopt.data.Dataset, got {type(data).__name__}"
            )
        self.data = data
        self.x, self.y = data.x, data.y
        self.input_names, self.output_names = data.input_names, data.output_names
        self.lower_bound, self.upper_bound = data.lower_bound, data.upper_bound
        self.batch_size = positive_int(batch_size, "batch_size")
        self.split = split_fractions(split)
        self.idx = split_indices(data.n_samples, self.split, random_state)

        y_train = self.y[self.idx["train"]]
        y_std = y_train.std(axis=0)
        for name, std in zip(self.output_names, y_std):
            if std == 0:
                raise ValueError(f"output {name!r} is constant on the training split")

        self.normalization = {
            "x_shift": ((self.upper_bound + self.lower_bound) / 2).tolist(),
            "x_scale": ((self.upper_bound - self.lower_bound) / 2).tolist(),
            "y_shift": y_train.mean(axis=0).tolist(),
            "y_scale": y_std.tolist(),
        }

    @property
    def n_inputs(self) -> int:
        return self.x.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.y.shape[1]

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
            Batches of normalized (X, Y) with shapes (batch_size, n_inputs) and (batch_size, n_outputs).
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
                dataset=TensorDataset(X_t, Y_t),
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
