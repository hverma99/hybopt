"""
A labeled data set (x, y) with column names and an input box, from arrays, DataFrames or CSV files.

Every surrogate type fits to a ``Dataset``.
Its input box becomes the input domain of the surrogate in the optimization model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from hybopt.data.split import split_indices


def as_arrays(x, y) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Check labeled data and convert it to float arrays.

    Inputs
    ------
    x : Inputs of shape (n, n_inputs), or (n,) for a single input; an array or a DataFrame.
    y : Outputs of shape (n,) or (n, n_outputs); an array, Series or DataFrame.

    Returns
    -------
    x, y          : Arrays of shape (n, n_inputs) and (n, n_outputs).
    input_names   : Column names of x (DataFrame columns, else "x1", "x2", ...).
    output_names  : Column names of y (DataFrame columns or Series name, else "y" or "y1", ...).
    """
    arrays = []
    for data, name in ((x, "x"), (y, "y")):
        try:  # copies, so later changes to data do not leak in
            if isinstance(data, (pd.DataFrame, pd.Series)):
                array = data.to_numpy(dtype=float, na_value=np.nan, copy=True)
            else:
                array = np.array(data, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be numeric, with the same number of columns in every row"
            ) from None
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1:] == (0,):
            raise ValueError(
                f"{name} must be 2-D (samples x columns), got shape {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or infinite values")
        arrays.append(array)
    x_array, y_array = arrays

    if len(x_array) != len(y_array):
        raise ValueError(
            f"x and y must have the same number of rows, got {len(x_array)} and {len(y_array)}"
        )
    if len(x_array) == 0:
        raise ValueError("x and y contain no samples")

    input_names = column_names(x, "x", x_array.shape[1])
    output_names = column_names(y, "y", y_array.shape[1])
    for names, name in ((input_names, "x"), (output_names, "y")):
        if len(set(names)) != len(names):
            raise ValueError(
                f"the columns of {name} must have unique names, got {names}"
            )
    if shared := sorted(set(input_names) & set(output_names)):
        raise ValueError(
            f"inputs and outputs must have different names, both have {shared}"
        )
    return x_array, y_array, input_names, output_names


def column_names(data, prefix: str, n_columns: int) -> list[str]:
    """
    Names of the columns of ``data``: DataFrame columns, a Series name, or prefix1, prefix2, ...
    """
    if isinstance(data, pd.DataFrame):
        return [str(c) for c in data.columns]
    if isinstance(data, pd.Series) and data.name is not None:
        return [str(data.name)]
    if n_columns == 1 and prefix == "y":
        return ["y"]
    return [f"{prefix}{i + 1}" for i in range(n_columns)]


def input_box(x: np.ndarray, lower_bound, upper_bound, names: list[str]):
    """
    The box [lower_bound, upper_bound] the inputs live in; by default the min and max of x.
    """
    if (lower_bound is None) != (upper_bound is None):
        raise ValueError("give both lower_bound and upper_bound, or neither")

    if lower_bound is None:
        lower, upper = x.min(axis=0), x.max(axis=0)
        for name, lo, hi in zip(names, lower, upper):
            if lo == hi:
                raise ValueError(
                    f"input {name!r} is constant ({lo:g}); remove it or give "
                    "lower_bound/upper_bound"
                )
        return lower, upper

    lower = np.asarray(lower_bound, dtype=float).ravel()
    upper = np.asarray(upper_bound, dtype=float).ravel()
    if len(lower) != x.shape[1] or len(upper) != x.shape[1]:
        raise ValueError(
            f"lower_bound and upper_bound need one value per input ({x.shape[1]}), "
            f"got {len(lower)} and {len(upper)}"
        )
    for name, lo, hi, column in zip(names, lower, upper, x.T):
        if not lo < hi:
            raise ValueError(
                f"input {name!r}: lower_bound {lo:g} must be below upper_bound {hi:g}"
            )
        if column.min() < lo or column.max() > hi:
            raise ValueError(
                f"input {name!r} has values in [{column.min():g}, {column.max():g}], "
                f"outside its bounds [{lo:g}, {hi:g}]"
            )
    return lower, upper


class Dataset:
    """
    A checked, labeled data set that any surrogate type can be fitted to.

    Inputs
    ------
    x                         : Inputs (n, n_inputs) or (n,); an array or a DataFrame.
    y                         : Outputs (n,) or (n, n_outputs); an array, Series or DataFrame.
    lower_bound, upper_bound  : Input box; default: the min and max of x.
    name                      : Label of the data set, e.g. "peaks" or a CSV file name.

    Attributes
    ----------
    x, y                      : Float arrays of shape (n, n_inputs) and (n, n_outputs).
    input_names, output_names : Column names (from DataFrames, else "x1", ... and "y" or "y1", ...).
    lower_bound, upper_bound  : Float arrays with the input box.
    """

    def __init__(self, x, y, lower_bound=None, upper_bound=None, name: str = "data"):

        self.x, self.y, self.input_names, self.output_names = as_arrays(x, y)

        self.lower_bound, self.upper_bound = input_box(
            self.x, lower_bound, upper_bound, self.input_names
        )
        self.name = name

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        inputs: list[str],
        outputs: list[str],
        lower_bound=None,
        upper_bound=None,
        name: str = "data",
    ) -> "Dataset":
        """
        Data set from the ``inputs`` and ``outputs`` columns of a DataFrame (other columns are ignored).
        """
        inputs, outputs = list(inputs), list(outputs)
        missing = [c for c in inputs + outputs if c not in frame.columns]
        if missing:
            raise ValueError(
                f"columns {missing} are not in the data; it has {list(frame.columns)}"
            )
        if overlap := sorted(set(inputs) & set(outputs)):
            raise ValueError(f"columns {overlap} are listed as both inputs and outputs")
        return cls(frame[inputs], frame[outputs], lower_bound, upper_bound, name)

    @classmethod
    def from_csv(
        cls,
        path: Path,
        inputs: list[str],
        outputs: list[str],
        lower_bound=None,
        upper_bound=None,
        name: str | None = None,
    ) -> "Dataset":
        """
        Data set from the ``inputs`` and ``outputs`` columns of a CSV file (one row per sample).
        The name defaults to the file name without extension.
        """
        frame = pd.read_csv(path, float_precision="round_trip")
        name = name if name is not None else Path(path).stem
        return cls.from_frame(frame, inputs, outputs, lower_bound, upper_bound, name)

    @property
    def n_samples(self) -> int:
        return self.x.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.x.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.y.shape[1]

    def split(self, split, seed: int = 0) -> dict[str, np.ndarray]:
        """
        Seeded train/val/test row indices, see ``split_indices``; the same for every surrogate type.
        """
        return split_indices(self.n_samples, split, seed)

    def subset(self, rows) -> "Dataset":
        """
        The data set restricted to ``rows`` (e.g. ``split(...)["train"]``), with the same input box.
        """
        frame = self.to_frame().iloc[rows]
        return Dataset(
            frame[self.input_names],
            frame[self.output_names],
            self.lower_bound,
            self.upper_bound,
            self.name,
        )

    def to_frame(self) -> pd.DataFrame:
        """
        The data as one DataFrame with the input columns followed by the output columns.
        """
        return pd.DataFrame(
            np.hstack([self.x, self.y]), columns=self.input_names + self.output_names
        )

    def __repr__(self) -> str:
        return (
            f"Dataset({self.name!r}: {self.n_samples} samples, inputs {self.input_names}, "
            f"outputs {self.output_names})"
        )
