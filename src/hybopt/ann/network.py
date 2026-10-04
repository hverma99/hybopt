"""
Feed-forward networks in PyTorch: build, map back to raw units, save and load.

A network is an ``nn.Sequential`` of Linear layers with the same activation (ReLU or tanh)
after every hidden layer and a linear output layer.
"""

import copy
import json
from pathlib import Path

import torch
from torch import nn

from hybopt.ann.checks import positive_int

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


def check_activation(activation) -> str:
    """
    Return ``activation`` if it is a supported activation ("relu" or "tanh").
    """
    if not isinstance(activation, str) or activation not in ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {sorted(ACTIVATIONS)}, got {activation!r}"
        )
    return activation


def build_model(
    network_in: int, widths: list[int], network_out: int, activation: str, seed: int
) -> nn.Sequential:
    """
    Build a dense network: Linear + activation per hidden layer, then a linear output layer.
    Weights are drawn with ``seed``: He-normal with zero biases for ReLU (PyTorch's default
    left shallow ReLU networks 2-3x less accurate), PyTorch's default for tanh.
    """
    network_in = positive_int(network_in, "network_in")
    network_out = positive_int(network_out, "network_out")
    widths = [positive_int(w, "every hidden-layer width") for w in widths]
    activation = check_activation(activation)

    torch.manual_seed(seed)

    sizes = [network_in, *widths]
    layers = []
    for n_from, n_to in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(n_from, n_to), ACTIVATIONS[activation]()]
    layers.append(nn.Linear(sizes[-1], network_out))

    if activation == "relu":
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    return nn.Sequential(*layers)


def fold_normalization(model: nn.Sequential, normalization: dict) -> nn.Sequential:
    """
    Return a copy of ``model`` that works on raw inputs and outputs.

    The model was trained on x_n = (x - x_shift) / x_scale and y_n = (y - y_shift) / y_scale.
    Folding these maps into the first and last Linear layers gives the same function on raw x and y,
    """
    raw = copy.deepcopy(model).double()

    linears = [m for m in raw if isinstance(m, nn.Linear)]
    first, last = linears[0], linears[-1]

    norm = {k: torch.tensor(v, dtype=torch.float64) for k, v in normalization.items()}

    with torch.no_grad():
        first.bias.sub_(first.weight @ (norm["x_shift"] / norm["x_scale"]))
        first.weight.div_(norm["x_scale"])
        last.weight.mul_(norm["y_scale"][:, None])
        last.bias.mul_(norm["y_scale"]).add_(norm["y_shift"])

    return raw


def save_model(model: nn.Sequential, activation: str, folder: Path, meta: dict) -> None:
    """
    Save the weights to ``model.pt`` and the architecture plus ``meta`` to ``meta.json``.
    """
    folder.mkdir(parents=True, exist_ok=True)

    linears = [m for m in model if isinstance(m, nn.Linear)]
    layer_sizes = [linears[0].in_features] + [m.out_features for m in linears]

    torch.save(model.state_dict(), folder / "model.pt")
    meta = {**meta, "activation": activation, "layer_sizes": layer_sizes}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))


def load_model(folder: Path) -> tuple[nn.Sequential, dict]:
    """
    Load a model saved by ``save_model``
    Returns the model and its meta.json.
    """
    meta = json.loads((folder / "meta.json").read_text())
    sizes = meta["layer_sizes"]

    model = build_model(sizes[0], sizes[1:-1], sizes[-1], meta["activation"], seed=0)
    model.double().load_state_dict(torch.load(folder / "model.pt"))
    return model.eval(), meta
