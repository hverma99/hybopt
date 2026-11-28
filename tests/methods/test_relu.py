from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import pytest
from scipy.optimize import check_grad

from hybopt.ann.network import build_model
from hybopt.bounds import interval_bounds
from hybopt.methods.relu import (
    add_bigm,
    bigm_ia,
    bigm_scaled,
    rescale,
    scale_relu,
    scaling_objective,
)
from hybopt.net import FeedForwardNet

ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = 1e-5  # MILP feasibility and integrality tolerances allow small deviations

# hidden neuron 0 is stably active (pre-activation in [4, 6]), neuron 1 stably inactive
# ([-11, -9]) and neuron 2 unstable ([-2, 2])
STABLE_AND_UNSTABLE = FeedForwardNet(
    weights=[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[1.0, 1.0, 1.0]]],
    biases=[[5.0, -10.0, 0.0], [0.0]],
    activation="relu",
    lower_bound=[-1.0, -1.0],
    upper_bound=[1.0, 1.0],
)


def random_net(widths: list[int], seed: int = 0) -> FeedForwardNet:
    model = build_model(2, widths, 1, "relu", seed=seed)
    return FeedForwardNet.from_torch(model, [-2.0, 0.0], [1.0, 3.0])


def sample_points(net: FeedForwardNet, n: int) -> np.ndarray:
    return np.random.default_rng(1).uniform(net.lower_bound, net.upper_bound, (n, net.n_inputs))


def hidden_outputs(net: FeedForwardNet, x: np.ndarray) -> list[np.ndarray]:
    """x^(k) of every hidden layer for one input point x."""
    outputs = []
    for w, b in zip(net.weights[:-1], net.biases[:-1]):
        x = np.maximum(w @ x + b, 0.0)
        outputs.append(x)
    return outputs


def solve_at(net: FeedForwardNet, x: np.ndarray, build=bigm_ia) -> pyo.ConcreteModel:
    """The encoding of ``net`` with its inputs fixed at x, solved with HiGHS."""
    m = pyo.ConcreteModel()
    m.nn = pyo.Block()
    build(m.nn, net)
    for i, value in enumerate(x):
        m.nn.x[0, i].fix(value)
    m.objective = pyo.Objective(expr=sum(m.nn.y.values()))  # the fixed inputs determine all
    result = pyo.SolverFactory("appsi_highs").solve(m)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    return m


def assert_encodes(net: FeedForwardNet, m: pyo.ConcreteModel, x: np.ndarray):
    """The solved model holds the network's hidden-layer outputs and outputs at x."""
    for k, values in enumerate(hidden_outputs(net, x), start=1):
        solved = [pyo.value(m.nn.x[k, i]) for i in range(len(values))]
        np.testing.assert_allclose(solved, values, atol=TOLERANCE)
    solved = [pyo.value(m.nn.y[i]) for i in range(net.n_outputs)]
    np.testing.assert_allclose(solved, net.predict(x)[0], atol=TOLERANCE)


@pytest.mark.parametrize("widths", [[4], [6, 5], [5, 5, 5]])
@pytest.mark.parametrize("seed", [0, 1])
def test_encodes_random_networks(widths, seed):
    net = random_net(widths, seed)
    for x in sample_points(net, 5):
        assert_encodes(net, solve_at(net, x), x)


def test_encodes_a_saved_network():
    net = FeedForwardNet.load(ROOT / "outputs/relu/models/peaks__relu__h2x10__t0")
    for x in sample_points(net, 5):
        assert_encodes(net, solve_at(net, x), x)


def test_stable_neurons_get_no_binary():
    net = STABLE_AND_UNSTABLE
    m = pyo.ConcreteModel()
    m.nn = pyo.Block()
    bigm_ia(m.nn, net)
    assert list(m.nn.z) == [(1, 2)]
    assert list(m.nn.stably_active) == [(1, 0)]
    assert list(m.nn.above_pre_activation) == list(m.nn.below_if_active) == [(1, 2)]
    assert list(m.nn.zero_if_inactive) == [(1, 2)]
    assert [m.nn.x[1, i].bounds for i in range(3)] == [(4.0, 6.0), (0.0, 0.0), (0.0, 2.0)]


@pytest.mark.parametrize("x", [[-1.0, -1.0], [1.0, 1.0], [0.3, -0.8], [-0.5, 0.9]])
def test_encodes_stable_and_unstable_neurons(x):
    x = np.array(x)
    assert_encodes(STABLE_AND_UNSTABLE, solve_at(STABLE_AND_UNSTABLE, x), x)


def test_one_binary_and_three_constraints_per_unstable_neuron():
    net = random_net([6, 5, 4])
    lower_upper = interval_bounds(net)[:-1]
    n_unstable = sum(np.sum((lower < 0) & (upper > 0)) for lower, upper in lower_upper)
    n_stably_active = sum(np.sum(lower >= 0) for lower, _ in lower_upper)
    m = pyo.ConcreteModel()
    m.nn = pyo.Block()
    bigm_ia(m.nn, net)
    assert n_unstable > 0
    assert len(m.nn.z) == len(m.nn.above_pre_activation) == n_unstable
    assert len(m.nn.below_if_active) == len(m.nn.zero_if_inactive) == n_unstable
    assert len(m.nn.stably_active) == n_stably_active
    assert len(m.nn.x) == 2 + 6 + 5 + 4 and len(m.nn.y) == len(m.nn.output) == 1
    assert len(m.nn.pre_activation) == 6 + 5 + 4 + 1


def test_inputs_and_outputs_are_bounded():
    net = random_net([6, 5])
    output_lower, output_upper = interval_bounds(net)[-1]
    m = pyo.ConcreteModel()
    m.nn = pyo.Block()
    bigm_ia(m.nn, net)
    assert [m.nn.x[0, i].bounds for i in range(2)] == [(-2.0, 1.0), (0.0, 3.0)]
    assert m.nn.y[0].bounds == (output_lower[0], output_upper[0])


def test_add_bigm_uses_the_given_bounds():
    net = random_net([6, 5])
    loose = [(lower - 10.0, upper + 10.0) for lower, upper in interval_bounds(net)]
    m = pyo.ConcreteModel()
    m.nn = pyo.Block()
    add_bigm(m.nn, net, loose)
    lower, upper = loose[0]
    assert [m.nn.x[1, i].bounds for i in range(6)] == [
        (max(lo, 0.0), max(up, 0.0)) for lo, up in zip(lower, upper)
    ]
    assert len(m.nn.z) == 11  # the loose bounds leave every neuron unstable

    def build_loose(block, net):
        add_bigm(block, net, loose)

    for x in sample_points(net, 3):
        assert_encodes(net, solve_at(net, x, build=build_loose), x)


# R3: a-posteriori scaling

SAVED_RELU_NETWORKS = sorted(path.parent for path in ROOT.glob("outputs/relu/models/*/meta.json"))
# y = 16 relu(3 x + 1) + 0.5: objective (12) is (3 + 1) c + 16 / c, smallest at c = 2
ONE_NEURON = FeedForwardNet([[[3.0]], [[16.0]]], [[1.0], [0.5]], "relu", [-1.0], [1.0])


def random_log_factors(net: FeedForwardNet, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 1.0, n) for n in net.layer_sizes[1:-1]]


def samples_and_corners(net: FeedForwardNet, n: int = 500) -> np.ndarray:
    axes = np.meshgrid(*zip(net.lower_bound, net.upper_bound))
    corners = np.array(axes).reshape(net.n_inputs, -1).T
    return np.vstack([sample_points(net, n), corners])


def assert_same_function(net: FeedForwardNet, other: FeedForwardNet):
    x = samples_and_corners(net)
    expected = net.predict(x)
    atol = 1e-10 * np.abs(expected).max()
    np.testing.assert_allclose(other.predict(x), expected, rtol=1e-10, atol=atol)


@pytest.mark.parametrize("widths", [[4], [6, 5], [5, 5, 5]])
@pytest.mark.parametrize("seed", [0, 1])
def test_rescale_keeps_the_function(widths, seed):
    net = random_net(widths, seed)
    assert_same_function(net, rescale(net, random_log_factors(net, seed)))


def test_rescale_by_zero_is_the_identity():
    net = random_net([6, 5])
    scaled = rescale(net, np.zeros(11))
    for w, scaled_w in zip(net.weights + net.biases, scaled.weights + scaled.biases):
        np.testing.assert_array_equal(w, scaled_w)
    assert scaled.input_names == net.input_names and scaled.output_names == net.output_names
    np.testing.assert_array_equal(scaled.lower_bound, net.lower_bound)


def test_rescale_checks_the_factors():
    net = random_net([6, 5])
    with pytest.raises(ValueError, match="needs 11 values, one per hidden neuron, got 3"):
        rescale(net, np.zeros(3))
    with pytest.raises(ValueError, match=r"sizes \[6, 5\], got sizes \[6, 4\]"):
        rescale(net, [np.zeros(6), np.zeros(4)])


def test_scaling_objective_is_the_absolute_sum_of_the_rescaled_network():
    net = random_net([6, 5, 4])
    factors = random_log_factors(net)
    scaled = rescale(net, factors)
    expected = sum(np.abs(w).sum() for w in scaled.weights)
    expected += sum(np.abs(b).sum() for b in scaled.biases[:-1])  # not the output bias
    value, _ = scaling_objective(net, np.concatenate(factors))
    assert value == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("widths", [[4], [6, 5], [5, 5, 5]])
def test_scaling_gradient_matches_finite_differences(widths):
    net = random_net(widths)
    point = np.concatenate(random_log_factors(net))
    error = check_grad(
        lambda c: scaling_objective(net, c)[0], lambda c: scaling_objective(net, c)[1], point
    )
    assert error <= 1e-5 * np.linalg.norm(scaling_objective(net, point)[1])


def test_hand_computed_scaling():
    scaled = scale_relu(ONE_NEURON)
    np.testing.assert_allclose(scaled.weights[0], [[6.0]], rtol=1e-5)
    np.testing.assert_allclose(scaled.biases[0], [2.0], rtol=1e-5)
    np.testing.assert_allclose(scaled.weights[1], [[8.0]], rtol=1e-5)
    np.testing.assert_array_equal(scaled.biases[1], [0.5])  # the output bias is not scaled


@pytest.mark.parametrize("folder", SAVED_RELU_NETWORKS, ids=lambda folder: folder.name)
def test_scaling_keeps_the_function_and_lowers_the_objective_of_saved_networks(folder):
    net = FeedForwardNet.load(folder)
    scaled = scale_relu(net)
    assert_same_function(net, scaled)
    n_factors = sum(net.layer_sizes[1:-1])
    unscaled = np.zeros(n_factors)
    assert scaling_objective(scaled, unscaled)[0] <= scaling_objective(net, unscaled)[0]


def test_interval_bounds_scale_with_the_neurons():
    # interval arithmetic commutes with positive scaling, so each neuron's bounds scale by its
    # factor and the output bounds do not change: scaling changes the magnitudes, not the relaxation
    net = random_net([6, 5, 4])
    factors = random_log_factors(net)
    scales = [np.exp(f) for f in factors] + [np.ones(net.n_outputs)]
    scaled_bounds = interval_bounds(rescale(net, factors))
    for (lower, upper), (scaled_lower, scaled_upper), c in zip(
        interval_bounds(net), scaled_bounds, scales
    ):
        np.testing.assert_allclose(scaled_lower, c * lower, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scaled_upper, c * upper, rtol=1e-12, atol=1e-12)


def test_scaling_keeps_dead_neurons_finite():
    # neuron 1 has no incoming weights or bias: its factor would go to infinity without the bounds
    weights = [[[1.0, -1.0], [0.0, 0.0]], [[2.0, 3.0]]]
    net = FeedForwardNet(weights, [[0.5, 0.0], [0.0]], "relu", [-1.0, -1.0], [1.0, 1.0])
    scaled = scale_relu(net, max_log_factor=10.0)
    assert all(np.all(np.isfinite(w)) for w in scaled.weights)
    assert scaled.weights[1][0, 1] == pytest.approx(3.0 * np.exp(-10.0))
    assert_same_function(net, scaled)


@pytest.mark.parametrize("function", [scale_relu, lambda net: rescale(net, [np.zeros(4)])])
def test_scaling_needs_a_relu_network(function):
    model = build_model(2, [4], 1, "tanh", seed=0)
    tanh = FeedForwardNet.from_torch(model, [-1.0, -1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="needs a relu network, got tanh"):
        function(tanh)


@pytest.mark.parametrize("widths", [[4], [6, 5], [5, 5, 5]])
@pytest.mark.parametrize("seed", [0, 1])
def test_scaled_encodes_random_networks(widths, seed):
    # the hidden variables hold the rescaled neurons, the inputs and outputs those of net
    net = random_net(widths, seed)
    scaled = scale_relu(net)
    for x in sample_points(net, 5):
        assert_encodes(scaled, solve_at(net, x, build=bigm_scaled), x)
        np.testing.assert_allclose(scaled.predict(x), net.predict(x), rtol=1e-10)


def test_scaled_encodes_a_saved_network():
    net = FeedForwardNet.load(ROOT / "outputs/relu/models/peaks__relu__h2x10__t0")
    scaled = scale_relu(net)
    for x in sample_points(net, 5):
        assert_encodes(scaled, solve_at(net, x, build=bigm_scaled), x)


def test_scaled_keeps_the_input_and_output_bounds():
    net = random_net([6, 5])
    m = pyo.ConcreteModel()
    m.nn = pyo.Block()
    bigm_scaled(m.nn, net)
    assert [m.nn.x[0, i].bounds for i in range(2)] == [(-2.0, 1.0), (0.0, 3.0)]
    output_lower, output_upper = interval_bounds(net)[-1]
    assert m.nn.y[0].bounds == pytest.approx((output_lower[0], output_upper[0]), rel=1e-12)
