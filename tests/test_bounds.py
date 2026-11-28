import itertools
from pathlib import Path

import numpy as np
import pytest

from hybopt.ann.network import build_model
from hybopt.bounds import interval_bounds, obbt_bounds
from hybopt.net import ACTIVATION_FUNCTIONS, FeedForwardNet

ROOT = Path(__file__).resolve().parents[1]
SAVED_NETWORKS = sorted(path.parent for path in ROOT.glob("outputs/*/models/*/meta.json"))

# y = 3 sigma(x1 - 2 x2 + 0.5) - 1 on [-1, 1] x [0, 2]: pre-activation of layer 1 in [-4.5, 1.5]
SMALL = {
    "weights": [[[1.0, -2.0]], [[3.0]]],
    "biases": [[0.5], [-1.0]],
    "lower_bound": [-1.0, 0.0],
    "upper_bound": [1.0, 2.0],
}


def pre_activations(net: FeedForwardNet, x: np.ndarray) -> list[np.ndarray]:
    """W^(k) x^(k-1) + b^(k) of every layer, each of shape (n, n_k)."""
    sigma = ACTIVATION_FUNCTIONS[net.activation]
    values = []
    for w, b in zip(net.weights, net.biases):
        values.append(x @ w.T + b)
        x = sigma(values[-1])
    return values


def random_net(activation: str, widths: list[int], seed: int = 0) -> FeedForwardNet:
    model = build_model(3, widths, 2, activation, seed=seed)
    return FeedForwardNet.from_torch(model, [-1.0, 0.0, 2.0], [1.0, 5.0, 3.0])


def assert_contains(bounds, values: list[np.ndarray]):
    """
    Every column of values[k] lies within bounds[k]. Interval arithmetic is exact at the box corners,
    where rounding in a different summation order may put a value a few ulps outside.
    """
    for (lower, upper), layer_values in zip(bounds, values):
        tolerance = 1e-12 * (1.0 + np.abs(layer_values).max())
        assert np.all(lower <= layer_values.min(axis=0) + tolerance)
        assert np.all(layer_values.max(axis=0) <= upper + tolerance)


def samples_and_corners(net: FeedForwardNet, n: int) -> np.ndarray:
    samples = np.random.default_rng(0).uniform(net.lower_bound, net.upper_bound, (n, net.n_inputs))
    corners = np.array(list(itertools.product(*zip(net.lower_bound, net.upper_bound))))
    return np.vstack([samples, corners])


def test_hand_computed_relu_bounds():
    bounds = interval_bounds(FeedForwardNet(**SMALL, activation="relu"))
    # layer 2 takes relu(layer 1) in [0, 1.5]: 3 * [0, 1.5] - 1
    expected = [([-4.5], [1.5]), ([-1.0], [3.5])]
    for (lower, upper), (expected_lower, expected_upper) in zip(bounds, expected):
        np.testing.assert_allclose(lower, expected_lower, rtol=1e-15)
        np.testing.assert_allclose(upper, expected_upper, rtol=1e-15)


def test_hand_computed_tanh_bounds():
    (lower_1, upper_1), (lower_2, upper_2) = interval_bounds(
        FeedForwardNet(**SMALL, activation="tanh")
    )
    np.testing.assert_allclose([lower_1[0], upper_1[0]], [-4.5, 1.5], rtol=1e-15)
    np.testing.assert_allclose(
        [lower_2[0], upper_2[0]], [3 * np.tanh(-4.5) - 1, 3 * np.tanh(1.5) - 1], rtol=1e-15
    )


def test_one_pair_per_layer():
    net = random_net("relu", [6, 5, 4])
    bounds = interval_bounds(net)
    assert [len(lower) for lower, _ in bounds] == [6, 5, 4, 2]
    assert [len(upper) for _, upper in bounds] == [6, 5, 4, 2]
    assert all(np.all(lower <= upper) for lower, upper in bounds)


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_first_layer_bounds_are_attained_at_box_corners(activation):
    # interval arithmetic is exact for an affine function of the box
    net = random_net(activation, [8, 4])
    corners = np.array(list(itertools.product(*zip(net.lower_bound, net.upper_bound))))
    first_layer = pre_activations(net, corners)[0]
    lower, upper = interval_bounds(net)[0]
    np.testing.assert_allclose(lower, first_layer.min(axis=0), rtol=1e-12)
    np.testing.assert_allclose(upper, first_layer.max(axis=0), rtol=1e-12)


@pytest.mark.parametrize("activation", ["relu", "tanh"])
@pytest.mark.parametrize("widths", [[5], [6, 5], [10, 10, 10, 10]])
@pytest.mark.parametrize("seed", [0, 1])
def test_bounds_contain_sampled_pre_activations(activation, widths, seed):
    net = random_net(activation, widths, seed)
    assert_contains(interval_bounds(net), pre_activations(net, samples_and_corners(net, 2000)))


@pytest.mark.parametrize("folder", SAVED_NETWORKS, ids=lambda folder: folder.name)
def test_bounds_contain_pre_activations_of_saved_networks(folder):
    net = FeedForwardNet.load(folder)
    assert_contains(interval_bounds(net), pre_activations(net, samples_and_corners(net, 2000)))


# y = relu(x) + relu(-x) on [-1, 1]: interval arithmetic gives y <= 2, the LP relaxation y <= 1
ABSOLUTE_VALUE = FeedForwardNet([[[1.0], [-1.0]], [[1.0, 1.0]]], [[0.0, 0.0], [0.0]], "relu", [-1.0], [1.0])
RELU_SAVED_NETWORKS = [
    folder
    for folder in SAVED_NETWORKS
    if folder.parts[-3] == "relu" and folder.name.split("__")[2] in ("h1x10", "h2x10", "h2x20", "h3x10")
]


def test_obbt_tightens_a_hand_computed_bound():
    (_, _), (lower, upper) = obbt_bounds(ABSOLUTE_VALUE)
    assert interval_bounds(ABSOLUTE_VALUE)[-1][1][0] == 2.0
    assert lower[0] == 0.0  # the interval bound is already tight
    assert upper[0] == pytest.approx(1.0, abs=2e-6)


def test_obbt_keeps_the_first_layer_and_is_never_looser():
    net = random_net("relu", [8, 6, 4])
    ia, obbt = interval_bounds(net), obbt_bounds(net)
    assert [len(lower) for lower, _ in obbt] == [8, 6, 4, 2]
    np.testing.assert_array_equal(obbt[0][0], ia[0][0])
    np.testing.assert_array_equal(obbt[0][1], ia[0][1])
    for (ia_lower, ia_upper), (lower, upper) in zip(ia, obbt):
        assert np.all(ia_lower <= lower) and np.all(upper <= ia_upper) and np.all(lower <= upper)


@pytest.mark.parametrize("widths", [[5], [6, 5], [10, 10, 10, 10]])
@pytest.mark.parametrize("seed", [0, 1])
def test_obbt_bounds_contain_sampled_pre_activations(widths, seed):
    net = random_net("relu", widths, seed)
    assert_contains(obbt_bounds(net), pre_activations(net, samples_and_corners(net, 2000)))


@pytest.mark.parametrize("folder", RELU_SAVED_NETWORKS, ids=lambda folder: folder.name)
def test_obbt_bounds_contain_pre_activations_of_saved_networks(folder):
    net = FeedForwardNet.load(folder)
    assert_contains(obbt_bounds(net), pre_activations(net, samples_and_corners(net, 2000)))


def test_obbt_with_a_non_persistent_solver():
    _, (_, upper) = obbt_bounds(ABSOLUTE_VALUE, solver="appsi_highs")
    assert upper[0] == pytest.approx(1.0, abs=2e-6)


def test_obbt_needs_a_relu_network():
    with pytest.raises(ValueError, match="obbt_bounds needs a relu network, got tanh"):
        obbt_bounds(random_net("tanh", [4]))


def test_obbt_unavailable_solver():
    with pytest.raises(ValueError, match="solver 'no_such_solver' is not available"):
        obbt_bounds(ABSOLUTE_VALUE, solver="no_such_solver")
