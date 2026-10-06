import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hybopt.ann import TrainSettings, fit_ann
from hybopt.ann.network import build_model, load_model
from hybopt.data import Dataset
from hybopt.net import FeedForwardNet

ROOT = Path(__file__).resolve().parents[1]
SAVED_NETWORKS = sorted(path.parent for path in ROOT.glob("outputs/*/models/*/meta.json"))

# |x1 - x2| = relu(x1 - x2) + relu(x2 - x1)
ABS_DIFF = {
    "weights": [[[1.0, -1.0], [-1.0, 1.0]], [[1.0, 1.0]]],
    "biases": [[0.0, 0.0], [0.0]],
    "activation": "relu",
    "lower_bound": [-5.0, -5.0],
    "upper_bound": [5.0, 5.0],
}


def torch_outputs(model: nn.Sequential, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float64)).numpy()


def points_in_box(lower, upper, n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(lower, upper, (n, len(lower)))


# --- evaluation -----------------------------------------------------------------------


def test_hand_built_relu_network():
    net = FeedForwardNet(**ABS_DIFF)
    x = np.array([[3.0, 1.0], [-2.0, 0.5], [0.0, 0.0], [-4.0, -4.5]])
    np.testing.assert_array_equal(net.predict(x), [[2.0], [2.5], [0.0], [0.5]])


def test_hand_built_tanh_network():
    # y = 3 tanh(2 x + 0.5) - tanh(-x) + 1
    net = FeedForwardNet(
        weights=[[[2.0], [-1.0]], [[3.0, -1.0]]],
        biases=[[0.5, 0.0], [1.0]],
        activation="tanh",
        lower_bound=[-2.0],
        upper_bound=[2.0],
    )
    x = np.linspace(-2, 2, 9)
    expected = 3 * np.tanh(2 * x + 0.5) - np.tanh(-x) + 1
    np.testing.assert_allclose(net.predict(x)[:, 0], expected, rtol=1e-13)


@pytest.mark.parametrize("activation", ["relu", "tanh"])
@pytest.mark.parametrize("widths", [[5], [6, 4], [10, 10, 10, 10]])
def test_from_torch_matches_the_torch_model(activation, widths):
    model = build_model(3, widths, 2, activation, seed=0).double()
    net = FeedForwardNet.from_torch(model, [-1.0, 0.0, 2.0], [1.0, 5.0, 3.0])
    x = points_in_box(net.lower_bound, net.upper_bound, 200)
    np.testing.assert_allclose(net.predict(x), torch_outputs(model, x), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_from_torch_reads_the_activation_and_architecture(activation):
    model = build_model(2, [7, 3], 1, activation, seed=0)
    net = FeedForwardNet.from_torch(model, [0.0, 0.0], [1.0, 1.0])
    assert net.activation == activation
    assert net.layer_sizes == [2, 7, 3, 1]
    assert (net.n_inputs, net.n_outputs) == (2, 1)
    assert all(w.dtype == np.float64 for w in net.weights + net.biases)


def test_from_torch_takes_a_float32_model():
    model = build_model(2, [4], 1, "relu", seed=0)  # float32, as before fold_normalization
    net = FeedForwardNet.from_torch(model, [0.0, 0.0], [1.0, 1.0])
    for w, m in zip(net.weights, [model[0], model[2]]):
        np.testing.assert_array_equal(w, m.weight.detach().double().numpy())


def test_from_torch_uses_zero_bias_for_linear_layers_without_bias():
    model = nn.Sequential(nn.Linear(2, 3, bias=False), nn.ReLU(), nn.Linear(3, 1))
    net = FeedForwardNet.from_torch(model, [0.0, 0.0], [1.0, 1.0])
    np.testing.assert_array_equal(net.biases[0], np.zeros(3))
    x = points_in_box(net.lower_bound, net.upper_bound, 20)
    np.testing.assert_allclose(net.predict(x), torch_outputs(model.double(), x), rtol=1e-12)


@pytest.mark.parametrize(
    "layers",
    [
        [nn.Linear(2, 3)],  # no hidden layer
        [nn.Linear(2, 3), nn.ReLU()],  # activation on the output
        [nn.Linear(2, 3), nn.Sigmoid(), nn.Linear(3, 1)],
        [nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 1)],
        [nn.Linear(2, 3), nn.Linear(3, 3), nn.Linear(3, 1)],
        [nn.Linear(2, 3), nn.ReLU(), nn.ReLU(), nn.Linear(3, 1)],
        [nn.Linear(2, 3), nn.Dropout(0.1), nn.Linear(3, 1)],
        [nn.Linear(2, 3), nn.ReLU(), nn.BatchNorm1d(3), nn.Linear(3, 1)],
    ],
)
def test_from_torch_rejects_other_architectures(layers):
    with pytest.raises(ValueError, match="model must alternate Linear layers and one activation"):
        FeedForwardNet.from_torch(nn.Sequential(*layers), [0.0, 0.0], [1.0, 1.0])


def test_predict_shapes():
    net = FeedForwardNet(**ABS_DIFF)
    assert net.predict([1.0, 2.0]).shape == (1, 1)  # one point
    assert net.predict(np.zeros((5, 2))).shape == (5, 1)
    single_input = FeedForwardNet(
        [[[1.0]], [[1.0], [2.0]]], [[0.0], [0.0, 0.0]], "relu", [0.0], [1.0]
    )
    assert single_input.predict([0.1, 0.2, 0.3]).shape == (3, 2)  # n points of one input


@pytest.mark.parametrize("x", [np.zeros((4, 3)), np.zeros((2, 2, 2)), [1.0, 2.0, 3.0]])
def test_predict_rejects_wrong_shapes(x):
    with pytest.raises(ValueError, match=r"x must have 2 columns \['x1', 'x2'\]"):
        FeedForwardNet(**ABS_DIFF).predict(x)


# --- loading saved networks -----------------------------------------------------------


def test_load_matches_the_saved_surrogate(tmp_path):
    x = np.random.default_rng(0).uniform(-1, 1, (300, 2))
    data = Dataset(x, np.sin(3 * x[:, 0]) * x[:, 1], [-1.0, -2.0], [1.0, 2.0])
    ann = fit_ann(
        data,
        split=(0.7, 0.15, 0.15),
        activation="tanh",
        depth=2,
        width=6,
        settings=TrainSettings(epochs=3),
        save_to=tmp_path / "m",
    )

    net = FeedForwardNet.load(tmp_path / "m")
    assert net.activation == "tanh" and net.layer_sizes == [2, 6, 6, 1]
    assert net.input_names == ["x1", "x2"] and net.output_names == ["y"]
    np.testing.assert_array_equal(net.lower_bound, [-1.0, -2.0])
    np.testing.assert_array_equal(net.upper_bound, [1.0, 2.0])
    points = points_in_box(net.lower_bound, net.upper_bound, 100)
    np.testing.assert_allclose(net.predict(points), ann.predict(points), rtol=1e-12, atol=1e-12)


def test_load_accepts_a_string_path(tmp_path):
    x = np.random.default_rng(0).uniform(0, 1, (100, 1))
    fit_ann(
        x,
        2 * x,
        split=(0.7, 0.15, 0.15),
        activation="relu",
        depth=1,
        width=3,
        settings=TrainSettings(epochs=1),
        save_to=tmp_path / "m",
    )
    assert FeedForwardNet.load(str(tmp_path / "m")).layer_sizes == [1, 3, 1]


@pytest.mark.parametrize("folder", SAVED_NETWORKS, ids=lambda folder: folder.name)
def test_saved_benchmark_networks_match_torch(folder):
    net = FeedForwardNet.load(folder)
    model, _ = load_model(folder)
    meta = json.loads((folder / "meta.json").read_text())
    assert net.layer_sizes == meta["layer_sizes"]
    assert net.activation == meta["activation"]
    np.testing.assert_array_equal(net.lower_bound, meta["domain"]["lower_bound"])
    np.testing.assert_array_equal(net.upper_bound, meta["domain"]["upper_bound"])
    x = points_in_box(net.lower_bound, net.upper_bound, 200)
    np.testing.assert_allclose(net.predict(x), torch_outputs(model, x), rtol=1e-12, atol=1e-12)


# --- checks ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weights, biases, message",
    [
        ([[[1.0, 1.0]]], [[0.0]], "at least one hidden layer"),
        ([[[1.0, 1.0]], [[1.0]]], [[0.0]], "one entry per layer, got 2 and 1"),
        ([[[1.0, 1.0]], [[1.0]]], [[0.0, 0.0], [0.0]], r"layer 1: weights must have shape \("),
        ([[1.0, 1.0], [[1.0]]], [[0.0], [0.0]], "layer 1: weights must have shape"),
        ([np.zeros((0, 2)), [[1.0]]], [[], [0.0]], "layer 1: weights must have shape"),
        ([[[1.0, 1.0]], [[1.0, 1.0]]], [[0.0], [0.0]], "layer 2 takes 2 inputs, but layer 1 has 1"),
        ([[[1.0, 1.0]], [[np.nan]]], [[0.0], [0.0]], "layer 2 has NaN or infinite"),
        ([[[1.0, 1.0]], [[1.0]]], [[np.inf], [0.0]], "layer 1 has NaN or infinite"),
        ([[[1.0, 1.0], [1.0]], [[1.0]]], [[0.0], [0.0]], "layer 1: weights and biases must be"),
    ],
)
def test_rejects_bad_layers(weights, biases, message):
    with pytest.raises(ValueError, match=message):
        FeedForwardNet(weights, biases, "relu", [0.0, 0.0], [1.0, 1.0])


@pytest.mark.parametrize("activation", ["sigmoid", "ReLU", None])
def test_rejects_unknown_activation(activation):
    with pytest.raises(ValueError, match="activation must be one of"):
        FeedForwardNet(**{**ABS_DIFF, "activation": activation})


@pytest.mark.parametrize(
    "lower, upper, message",
    [
        ([0.0], [1.0, 1.0], r"one value per input \(2\), got 1 and 2"),
        ([0.0, 1.0], [1.0, 1.0], "input 'x2': the box needs finite bounds"),
        ([0.0, 2.0], [1.0, 1.0], "lower_bound < upper_bound, got \\[2, 1\\]"),
        ([0.0, -np.inf], [1.0, 1.0], "input 'x2'"),
        ([0.0, 0.0], [np.nan, 1.0], "input 'x1'"),
    ],
)
def test_rejects_bad_box(lower, upper, message):
    with pytest.raises(ValueError, match=message):
        FeedForwardNet(**{**ABS_DIFF, "lower_bound": lower, "upper_bound": upper})


def test_default_and_given_names():
    assert FeedForwardNet(**ABS_DIFF).input_names == ["x1", "x2"]
    assert FeedForwardNet(**ABS_DIFF).output_names == ["y"]
    two_outputs = FeedForwardNet(
        **{
            **ABS_DIFF,
            "weights": [ABS_DIFF["weights"][0], [[1.0, 1.0], [1.0, 0.0]]],
            "biases": [[0.0, 0.0], [0.0, 0.0]],
        }
    )
    assert two_outputs.output_names == ["y1", "y2"]
    named = FeedForwardNet(**ABS_DIFF, input_names=("T", "tau"), output_names=["X"])
    assert named.input_names == ["T", "tau"] and named.output_names == ["X"]


@pytest.mark.parametrize(
    "input_names, output_names, message",
    [
        (["a"], None, r"input_names needs 2 names, got \['a'\]"),
        (["a", "a"], None, "input_names must be unique"),
        (None, ["y", "z"], "output_names needs 1 names"),
        (["a", "y"], ["y"], r"inputs and outputs must have different names, both have \['y'\]"),
    ],
)
def test_rejects_bad_names(input_names, output_names, message):
    with pytest.raises(ValueError, match=message):
        FeedForwardNet(**ABS_DIFF, input_names=input_names, output_names=output_names)


def test_keeps_copies_of_the_arrays():
    weights = [np.array(w) for w in ABS_DIFF["weights"]]
    lower = np.array(ABS_DIFF["lower_bound"])
    net = FeedForwardNet(weights, ABS_DIFF["biases"], "relu", lower, ABS_DIFF["upper_bound"])
    weights[0][0, 0] = 100.0
    lower[0] = -100.0
    assert net.weights[0][0, 0] == 1.0 and net.lower_bound[0] == -5.0


def test_repr():
    assert repr(FeedForwardNet(**ABS_DIFF)) == (
        "FeedForwardNet(relu, layer sizes [2, 2, 1], inputs ['x1', 'x2'], outputs ['y'])"
    )
