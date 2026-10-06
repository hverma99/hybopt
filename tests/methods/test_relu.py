from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import pytest

from hybopt.ann.network import build_model
from hybopt.bounds import interval_bounds
from hybopt.methods.relu import add_bigm, bigm_ia
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
