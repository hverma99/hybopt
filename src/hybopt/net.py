"""
A trained feed-forward network as float64 NumPy arrays, the form every embedding method works on.

With K hidden layers and the same activation sigma (ReLU or tanh) in each, the network maps raw
inputs x in the box [lower_bound, upper_bound] to raw outputs y:

    x^0 = x,    x^k = sigma(W^k x^(k-1) + b^k)  for k = 1, ..., K,    y = W^(K+1) x^K + b^(K+1)

Typical use:

    net = FeedForwardNet.load("outputs/relu/models/peaks__relu__h2x10__t0")  # ANNSurrogate.save
    net = FeedForwardNet.from_torch(ann.model, ann.lower_bound, ann.upper_bound)  # from fit_ann
    net = FeedForwardNet(weights, biases, "relu", lower_bound, upper_bound)  # raw arrays
"""

from pathlib import Path

import numpy as np
from torch import nn

from hybopt.ann.network import ACTIVATIONS, check_activation, load_model
from hybopt.data.dataset import column_names

ACTIVATION_FUNCTIONS = {"relu": lambda z: np.maximum(z, 0.0), "tanh": np.tanh}


def check_layers(weights, biases) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Copy the weights and biases to float64 arrays and check that consecutive layers fit together.
    """
    if len(weights) != len(biases):
        raise ValueError(
            f"weights and biases need one entry per layer, got {len(weights)} and {len(biases)}"
        )
    if len(weights) < 2:
        raise ValueError(
            "the network needs at least one hidden layer and an output layer, "
            f"got {len(weights)} layers"
        )

    weight_arrays, bias_arrays = [], []
    for k, (w, b) in enumerate(zip(weights, biases), start=1):
        try:  # copies, so later changes to the caller's arrays do not leak in
            w, b = np.array(w, dtype=float), np.array(b, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"layer {k}: weights and biases must be numeric arrays") from None
        if w.ndim != 2 or 0 in w.shape or b.shape != (w.shape[0],):
            raise ValueError(
                f"layer {k}: weights must have shape (n_out, n_in) and biases (n_out,), "
                f"got {w.shape} and {b.shape}"
            )
        if weight_arrays and w.shape[1] != weight_arrays[-1].shape[0]:
            raise ValueError(
                f"layer {k} takes {w.shape[1]} inputs, but layer {k - 1} has "
                f"{weight_arrays[-1].shape[0]} outputs"
            )
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(b))):
            raise ValueError(f"layer {k} has NaN or infinite weights or biases")
        weight_arrays.append(w)
        bias_arrays.append(b)
    return weight_arrays, bias_arrays


def check_names(names, prefix: str, n_names: int, label: str) -> list[str]:
    """
    ``names`` as a list of n_names unique strings; by default prefix1, prefix2, ... (or "y").
    """
    if names is None:
        return column_names(None, prefix, n_names)
    names = [str(name) for name in names]
    if len(names) != n_names:
        raise ValueError(f"{label} needs {n_names} names, got {names}")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must be unique, got {names}")
    return names


def check_box(lower_bound, upper_bound, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    The input box as float arrays, with finite bounds and lower_bound < upper_bound for every input.
    """
    lower = np.array(lower_bound, dtype=float).ravel()
    upper = np.array(upper_bound, dtype=float).ravel()
    if len(lower) != len(names) or len(upper) != len(names):
        raise ValueError(
            f"lower_bound and upper_bound need one value per input ({len(names)}), "
            f"got {len(lower)} and {len(upper)}"
        )
    for name, lo, hi in zip(names, lower, upper):
        if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
            raise ValueError(
                f"input {name!r}: the box needs finite bounds with lower_bound < upper_bound, "
                f"got [{lo:g}, {hi:g}]"
            )
    return lower, upper


def sequential_activation(model: nn.Sequential) -> str:
    """
    The hidden activation of ``model`` ("relu" or "tanh"), after checking that it alternates
    Linear layers and one activation type, with at least one hidden layer and a Linear output layer.
    """
    names = {module: name for name, module in ACTIVATIONS.items()}
    layers = list(model)
    activations = {type(m) for m in layers[1::2]}
    is_dense = (
        len(layers) >= 3
        and len(layers) % 2 == 1
        and all(type(m) is nn.Linear for m in layers[::2])
        and len(activations) == 1
        and activations <= names.keys()
    )
    if not is_dense:
        raise ValueError(
            "model must alternate Linear layers and one activation (nn.ReLU or nn.Tanh), with "
            "at least one hidden layer and a Linear output layer, "
            f"got {[type(m).__name__ for m in layers]}"
        )
    return names[activations.pop()]


class FeedForwardNet:
    """
    A dense feed-forward network in raw units, with the input box it is embedded on.

    Inputs
    ------
    weights                   : W^1, ..., W^(K+1); W^k has shape (n_k, n_(k-1)), as in nn.Linear.
    biases                    : b^1, ..., b^(K+1); b^k has shape (n_k,).
    activation                : "relu" or "tanh" in every hidden layer (the output layer is linear).
    lower_bound, upper_bound  : Input box, one finite value per input.
    input_names, output_names : Input and output names; default "x1", ... and "y" or "y1", ...

    Attributes
    ----------
    weights, biases           : Lists of float64 arrays (copies of the inputs).
    lower_bound, upper_bound  : Float arrays with the input box.
    activation, input_names, output_names : As given (names as lists of strings).
    """

    def __init__(
        self,
        weights,
        biases,
        activation: str,
        lower_bound,
        upper_bound,
        input_names=None,
        output_names=None,
    ):
        self.weights, self.biases = check_layers(weights, biases)
        self.activation = check_activation(activation)
        self.input_names = check_names(input_names, "x", self.n_inputs, "input_names")
        self.output_names = check_names(output_names, "y", self.n_outputs, "output_names")
        if shared := sorted(set(self.input_names) & set(self.output_names)):
            raise ValueError(f"inputs and outputs must have different names, both have {shared}")
        self.lower_bound, self.upper_bound = check_box(lower_bound, upper_bound, self.input_names)

    @classmethod
    def from_torch(
        cls,
        model: nn.Sequential,
        lower_bound,
        upper_bound,
        input_names=None,
        output_names=None,
    ) -> "FeedForwardNet":
        """
        Network from an ``nn.Sequential`` on raw units, such as ``ANNSurrogate.model``; the
        activation is read from the model.
        """
        activation = sequential_activation(model)
        linears = list(model)[::2]
        return cls(
            weights=[m.weight.detach().double().numpy() for m in linears],
            biases=[
                np.zeros(m.out_features) if m.bias is None else m.bias.detach().double().numpy()
                for m in linears
            ],
            activation=activation,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            input_names=input_names,
            output_names=output_names,
        )

    @classmethod
    def load(cls, folder: Path) -> "FeedForwardNet":
        """
        Network saved by ``ANNSurrogate.save`` (model.pt + meta.json), its domain as the input box.
        """
        model, meta = load_model(Path(folder))
        return cls.from_torch(
            model,
            meta["domain"]["lower_bound"],
            meta["domain"]["upper_bound"],
            meta["input_names"],
            meta["output_names"],
        )

    @property
    def layer_sizes(self) -> list[int]:
        """
        [n_inputs, hidden widths ..., n_outputs], as in meta.json.
        """
        return [self.n_inputs] + [w.shape[0] for w in self.weights]

    @property
    def n_inputs(self) -> int:
        return self.weights[0].shape[1]

    @property
    def n_outputs(self) -> int:
        return self.weights[-1].shape[0]

    def predict(self, x) -> np.ndarray:
        """
        Outputs, shape (n, n_outputs), for inputs x of shape (n, n_inputs). A 1-D x is one point,
        or n points when the network has a single input (as in ``ANNSurrogate.predict``).
        """
        x = np.array(x, dtype=float)
        if x.ndim == 1:
            x = x[:, None] if self.n_inputs == 1 else x[None, :]
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(
                f"x must have {self.n_inputs} columns {self.input_names}, got shape {x.shape}"
            )
        sigma = ACTIVATION_FUNCTIONS[self.activation]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = sigma(x @ w.T + b)
        return x @ self.weights[-1].T + self.biases[-1]

    def __repr__(self) -> str:
        return (
            f"FeedForwardNet({self.activation}, layer sizes {self.layer_sizes}, "
            f"inputs {self.input_names}, outputs {self.output_names})"
        )
