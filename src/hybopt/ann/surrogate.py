"""
Fit an ANN surrogate to any labeled data set, evaluate it, and save it for embedding.

Typical use, with any labeled data (arrays or DataFrames) or a ``hybopt.data.Dataset``:

    ann = fit_ann(x, y, split=(0.7, 0.15, 0.15), activation="tanh", depth=2, width=20)
    ann = fit_ann(Dataset.from_csv("runs.csv", inputs=["T", "tau"], outputs=["X"]), split=..., ...)
    ann.metrics["test"]["rel_rmse"]          # errors on the train/val/test splits
    ann.save(Path("outputs/models/my_model"))  # model.pt + meta.json + history.csv

The benchmark functions (package ``benchmarks``) are just one data source:

    data = sample_dataset(get_function("peaks"), 10000, seed=0)   # input box = the function's domain
    fit_ann(data, split=(0.7, 0.15, 0.15), activation="relu", depth=2, width=20)
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from hybopt.ann.checks import positive_int
from hybopt.ann.data import DataModule
from hybopt.ann.network import (
    build_model,
    check_activation,
    fold_normalization,
    load_model,
    save_model,
)
from hybopt.ann.train import TrainSettings, error_metrics, train_network
from hybopt.data import Dataset


@dataclass
class ANNSurrogate:
    """
    A trained ANN surrogate in raw units, with what is needed to embed it in an optimization model.

    model                     : float64 network on raw inputs and outputs (scaling folded in).
    activation                : "relu" or "tanh".
    input_names, output_names : Column names of x and y.
    lower_bound, upper_bound  : Input box of the model (the domain in the optimization model).
    normalization             : Input/output scaling used in training, already folded into model.
    metrics                   : Errors on the train/val/test splits, see ``error_metrics``.
    history                   : Per-epoch training and validation loss (normalized outputs).
    training                  : What produced the model: depth, width, split, seeds, settings, ...
    """

    model: nn.Sequential
    activation: str
    input_names: list[str]
    output_names: list[str]
    lower_bound: list[float]
    upper_bound: list[float]
    normalization: dict
    metrics: dict
    history: list[dict]
    training: dict

    def predict(self, x):
        """
        Outputs for inputs x of shape (n, n_inputs).

        An array gives an array of shape (n, n_outputs). A DataFrame is matched to the inputs by
        column name (in any order) and gives a DataFrame with the output names and its index.
        """
        if isinstance(x, pd.DataFrame):
            missing = [name for name in self.input_names if name not in x.columns]
            if missing:
                raise ValueError(
                    f"x is missing the input columns {missing}; pass an array to match "
                    "inputs by position instead"
                )
            y = self.predict(x[self.input_names].to_numpy(dtype=float))
            return pd.DataFrame(y, columns=self.output_names, index=x.index)

        x = np.ascontiguousarray(
            x, dtype=float
        )  # torch rejects views such as reordered columns
        if x.ndim == 1:
            x = x[:, None] if len(self.input_names) == 1 else x[None, :]
        if x.ndim != 2 or x.shape[1] != len(self.input_names):
            raise ValueError(
                f"x must have {len(self.input_names)} columns {self.input_names}, "
                f"got shape {x.shape}"
            )
        with torch.no_grad():
            return self.model(torch.tensor(x, dtype=torch.float64)).numpy()

    def save(
        self, folder: Path, meta: dict | None = None, overwrite: bool = False
    ) -> None:
        """
        Save to ``folder``: model.pt (weights), meta.json (scaling, bounds, errors, ...) and
        history.csv. Extra entries in ``meta`` are added to meta.json. An existing model is
        only replaced when ``overwrite`` is True.
        """
        folder = Path(folder)
        if (folder / "model.pt").exists() and not overwrite:
            raise FileExistsError(
                f"{folder} already holds a model; pass overwrite=True to replace it"
            )
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.history, columns=["epoch", "train_mse", "val_mse"]).to_csv(
            folder / "history.csv", index=False, float_format="%.17g"
        )

        # meta.json is written last, so a folder with meta.json holds a complete model
        info = {
            "input_names": self.input_names,
            "output_names": self.output_names,
            "domain": {
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
            },
            "normalization": self.normalization,
            "metrics": self.metrics,
            "training": self.training,
        }
        save_model(self.model, self.activation, folder, meta={**(meta or {}), **info})

    @classmethod
    def load(cls, folder: Path) -> "ANNSurrogate":
        """
        Load a surrogate saved by ``save``.
        """
        folder = Path(folder)
        model, meta = load_model(folder)
        history = pd.read_csv(folder / "history.csv", float_precision="round_trip")
        return cls(
            model=model,
            activation=meta["activation"],
            input_names=meta["input_names"],
            output_names=meta["output_names"],
            lower_bound=meta["domain"]["lower_bound"],
            upper_bound=meta["domain"]["upper_bound"],
            normalization=meta["normalization"],
            metrics=meta["metrics"],
            history=history.to_dict("records"),
            training=meta["training"],
        )


def fit_ann(
    x,
    y=None,
    *,
    split,
    activation: str,
    depth: int,
    width: int,
    lower_bound=None,
    upper_bound=None,
    seed: int = 0,
    split_seed: int = 0,
    settings: TrainSettings | None = None,
    save_to: Path | None = None,
    meta: dict | None = None,
    overwrite: bool = False,
) -> ANNSurrogate:
    """
    Fit an ANN surrogate to labeled data, evaluate it on train/val/test, and optionally save it.

    Inputs
    ------
    x                         : A ``Dataset``, or inputs (n, n_inputs) or (n,) as an array or DataFrame.
    y                         : Outputs (n,) or (n, n_outputs) as an array, Series or DataFrame;
                                leave out when x is a Dataset.
    split                     : Fractions {"train", "val", "test"} or (train, val, test), summing to 1.
    activation                : "relu" or "tanh" (used in every hidden layer).
    depth                     : Number of hidden layers.
    width                     : Neurons per hidden layer.
    lower_bound, upper_bound  : Input box for x, y; default: min and max of x (a Dataset brings
                                its own). Becomes the model's domain in the optimization model.
    seed                      : Seed for weight initialization and batch shuffling.
    split_seed                : Seed for the train/val/test split.
    settings                  : TrainSettings (learning rate, batch size, epochs, patience).
    save_to                   : Folder to save the model in (optional), see ``ANNSurrogate.save``.
    meta                      : Extra entries for meta.json when saving.
    overwrite                 : Replace a model already saved in ``save_to``.

    Returns
    -------
    ANNSurrogate : trained model in raw units, its scaling and input box, errors and history.

    split, activation, depth, width and the later arguments are keyword-only. Every random step is
    seeded, so a call is reproducible for a fixed number of torch threads (the scripts use one).
    """
    if isinstance(x, Dataset):
        if y is not None or lower_bound is not None or upper_bound is not None:
            raise ValueError(
                "with a Dataset, leave out y, lower_bound and upper_bound; the Dataset has them"
            )
        data = x
    elif y is None:
        raise ValueError("give the outputs y, or pass a Dataset as the first argument")
    else:
        data = Dataset(x, y, lower_bound, upper_bound)
    activation = check_activation(activation)
    depth = positive_int(depth, "depth")
    width = positive_int(width, "width")
    settings = settings if settings is not None else TrainSettings()
    if save_to is not None and (Path(save_to) / "model.pt").exists() and not overwrite:
        raise FileExistsError(
            f"{save_to} already holds a model; pass overwrite=True to replace it"
        )

    dm = DataModule(
        data, batch_size=settings.batch_size, split=split, random_state=split_seed
    )

    start = time.perf_counter()
    model = build_model(dm.n_inputs, [width] * depth, dm.n_outputs, activation, seed)
    train_loader, val_loader, _ = dm.labeled_data_loader(shuffle_seed=seed)
    history = train_network(model, train_loader, val_loader, settings)
    seconds = time.perf_counter() - start

    raw_model = fold_normalization(model, dm.normalization)
    surrogate = ANNSurrogate(
        model=raw_model,
        activation=activation,
        input_names=dm.input_names,
        output_names=dm.output_names,
        lower_bound=dm.lower_bound.tolist(),
        upper_bound=dm.upper_bound.tolist(),
        normalization=dm.normalization,
        metrics=error_metrics(raw_model, dm),
        history=history,
        training={
            "data": data.name,
            "depth": depth,
            "width": width,
            "split": dm.split,
            "split_seed": split_seed,
            "seed": seed,
            "settings": asdict(settings),
            "n_samples": {name: len(idx) for name, idx in dm.idx.items()},
            "epochs_run": len(history),
            "best_epoch": min(history, key=lambda h: h["val_mse"])["epoch"],
            "train_seconds": round(seconds, 2),
        },
    )

    if save_to is not None:
        surrogate.save(save_to, meta=meta, overwrite=overwrite)
    return surrogate
