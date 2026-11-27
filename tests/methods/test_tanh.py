from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import pytest

from hybopt.ann.network import build_model
from hybopt.bounds import interval_bounds
from hybopt.methods.tanh import (
    MAX_CURVATURE,
    XSTAR,
    Piece,
    big_m,
    curvature_range,
    evaluate,
    pq,
    tanh_pieces,
    tanh_second_derivative,
)
from hybopt.net import FeedForwardNet
from hybopt.solve import minimize_output

ROOT = Path(__file__).resolve().parents[2]
ROUNDING = 1e-12  # the enclosure is exact up to floating-point rounding
TOLERANCE = 1e-5  # solver feasibility and integrality tolerances


def random_intervals(n: int, seed: int = 0) -> list[tuple[float, float]]:
    """Random intervals of very different widths and positions, plus some around -x*, 0 and x*."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-4, 4, n)
    widths = 10.0 ** rng.uniform(-3, 1.3, n)
    fixed = [(-XSTAR - 0.1, -XSTAR + 0.1), (XSTAR - 0.5, XSTAR + 0.5), (-0.2, 0.3), (-20.0, 20.0)]
    return fixed + [(c - w / 2, c + w / 2) for c, w in zip(centers, widths)]


def all_pieces(tol: float = 0.02, max_pieces: int = 12) -> list[Piece]:
    return [p for l, u in random_intervals(40) for p in tanh_pieces(l, u, tol, max_pieces)]


def random_net(widths: list[int], seed: int = 0) -> FeedForwardNet:
    model = build_model(2, widths, 1, "tanh", seed=seed)
    return FeedForwardNet.from_torch(model, [-2.0, 0.0], [1.0, 3.0])


def sample_points(net: FeedForwardNet, n: int) -> np.ndarray:
    return np.random.default_rng(1).uniform(net.lower_bound, net.upper_bound, (n, net.n_inputs))


def build(net: FeedForwardNet, **options) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()
    m.nn = pyo.Block()
    pq(m.nn, net, **options)
    return m


def test_curvature_range_is_exact():
    assert tanh_second_derivative(-XSTAR) == pytest.approx(MAX_CURVATURE)
    assert tanh_second_derivative(XSTAR) == pytest.approx(-MAX_CURVATURE)
    for a, b in random_intervals(40):
        t = np.linspace(a, b, 20001)
        values = tanh_second_derivative(t)
        c_min, c_max = curvature_range(a, b)
        assert c_min <= values.min() + ROUNDING and values.max() <= c_max + ROUNDING
        assert c_min == pytest.approx(values.min(), abs=1e-6)
        assert c_max == pytest.approx(values.max(), abs=1e-6)


def test_quadratics_interpolate_tanh_at_the_endpoints():
    for p in all_pieces():
        for coefficients in (p.lower, p.upper):
            ends = np.array([p.a, p.b])
            np.testing.assert_allclose(evaluate(coefficients, ends), np.tanh(ends), atol=ROUNDING)


def test_quadratics_enclose_tanh():
    for p in all_pieces():
        x = np.linspace(p.a, p.b, 501)
        assert np.all(evaluate(p.lower, x) <= np.tanh(x) + ROUNDING)
        assert np.all(np.tanh(x) <= evaluate(p.upper, x) + ROUNDING)


def test_quadratics_are_convex_and_pieces_one_signed():
    for p in all_pieces():
        assert p.kappa_up <= 0.0 <= p.kappa_lo
        assert p.b <= 0.0 or p.a >= 0.0  # no piece crosses 0 (the intervals are not near 0)


def test_width_closed_form_matches_the_sampled_band():
    for p in all_pieces():
        x = np.linspace(p.a, p.b, 2001)  # odd count: the midpoint is sampled
        sampled = (evaluate(p.upper, x) - evaluate(p.lower, x)).max()
        assert p.width == pytest.approx(sampled, rel=1e-6, abs=1e-12)
        assert p.width <= (p.b - p.a) ** 2 / (6 * np.sqrt(3)) + ROUNDING


@pytest.mark.parametrize("a, b", [(-3.0, 0.0), (0.0, 2.5), (0.2, 0.9), (-1.7, -0.1)])
def test_bisection_shrinks_the_band_by_four(a, b):
    parent = Piece.make(a, b)
    middle = (a + b) / 2
    for child in (Piece.make(a, middle), Piece.make(middle, b)):
        assert child.width <= parent.width / 4 + ROUNDING


def test_pieces_cover_the_interval_and_meet_the_tolerance():
    for l, u in random_intervals(40):
        pieces = tanh_pieces(l, u, tol=0.01, max_pieces=200)
        assert pieces[0].a == l and pieces[-1].b == u
        assert all(p.b == q.a for p, q in zip(pieces[:-1], pieces[1:]))
        assert max(p.width for p in pieces) <= 0.01
        if l < 0 < u:
            assert 0.0 in [p.a for p in pieces]


def test_large_tolerance_gives_the_initial_breakpoints():
    assert [(p.a, p.b) for p in tanh_pieces(-2.0, 3.0, tol=10.0)] == [(-2.0, 0.0), (0.0, 3.0)]
    assert [(p.a, p.b) for p in tanh_pieces(0.5, 3.0, tol=10.0)] == [(0.5, 3.0)]


def test_one_piece_is_a_single_band():
    (piece,) = tanh_pieces(-2.0, 3.0, max_pieces=1)
    assert (piece.a, piece.b) == (-2.0, 3.0)
    x = np.linspace(-2.0, 3.0, 1001)
    assert np.all(evaluate(piece.lower, x) <= np.tanh(x) + ROUNDING)
    assert np.all(np.tanh(x) <= evaluate(piece.upper, x) + ROUNDING)


@pytest.mark.parametrize("l, u", [(-5.0, 4.0), (0.1, 6.0), (-0.9, -0.3)])
def test_smaller_tolerance_refines_the_breakpoints(l, u):
    coarse = {p.a for p in tanh_pieces(l, u, tol=0.05, max_pieces=100)}
    fine = {p.a for p in tanh_pieces(l, u, tol=0.001, max_pieces=100)}
    assert coarse < fine


def test_degenerate_interval_is_padded():
    (piece,) = tanh_pieces(0.3, 0.3)
    assert piece.a < 0.3 < piece.b


@pytest.mark.parametrize("tol, max_pieces", [(0.0, 12), (-1.0, 12), (0.02, 0), (0.02, -2.5), (0.02, 2.5)])
def test_rejects_bad_options(tol, max_pieces):
    with pytest.raises(ValueError):
        tanh_pieces(-1.0, 1.0, tol, max_pieces)


def test_big_m_makes_other_pieces_inactive():
    l, u = -4.0, 3.0
    pieces = tanh_pieces(l, u, tol=0.001, max_pieces=30)
    x = np.linspace(l, u, 2001)
    for p in pieces:
        m_lo, m_up = big_m(p, l, u)
        assert np.all(evaluate(p.lower, x) - m_lo <= np.tanh(l) + ROUNDING)
        assert np.all(evaluate(p.upper, x) + m_up >= np.tanh(u) - ROUNDING)


def true_point(m: pyo.ConcreteModel, net: FeedForwardNet, x: np.ndarray) -> None:
    """Set the model's variables to the network's values at x, each z to the piece holding it."""
    for i, value in enumerate(x):
        m.nn.x[0, i].set_value(value)
    for k, (w, b) in enumerate(zip(net.weights[:-1], net.biases[:-1]), start=1):
        pre = w @ x + b
        x = np.tanh(pre)
        for i in range(len(pre)):
            m.nn.pre_activation[k, i].set_value(pre[i])
            m.nn.x[k, i].set_value(x[i])
            pieces = m.nn.tanh_pieces[k, i]
            chosen = next(s for s, p in enumerate(pieces) if p.a <= pre[i] <= p.b)
            for s in range(len(pieces)):
                m.nn.z[k, i, s].set_value(int(s == chosen))
    for i, value in enumerate(net.weights[-1] @ x + net.biases[-1]):
        m.nn.y[i].set_value(value)


@pytest.mark.parametrize("widths", [[5], [4, 4], [3, 3, 3]])
def test_true_values_are_feasible(widths):
    net = random_net(widths)
    m = build(net)
    for x in sample_points(net, 20):
        true_point(m, net, x)
        for c in m.component_data_objects(pyo.Constraint, active=True):
            assert c.lslack() >= -1e-9 and c.uslack() >= -1e-9, c.name
        for v in m.component_data_objects(pyo.Var):
            assert v.lb - 1e-9 <= v.value <= v.ub + 1e-9, v.name


def test_model_structure():
    net = random_net([4, 3])
    m = build(net, tol=0.01)
    n_pieces = {n: len(p) for n, p in m.nn.tanh_pieces.items()}
    assert sorted(n_pieces) == [(1, i) for i in range(4)] + [(2, i) for i in range(3)]
    assert len(m.nn.z) == len(m.nn.above_lower_quadratic) == sum(n_pieces.values())
    assert len(m.nn.below_upper_quadratic) == sum(n_pieces.values())
    assert len(m.nn.one_piece) == len(m.nn.linear) == len(m.nn.pre_activation) == 7
    assert len(m.nn.x) == 2 + 4 + 3 and len(m.nn.y) == len(m.nn.output) == 1
    lower, upper = interval_bounds(net)[0]
    assert [m.nn.pre_activation[1, i].bounds for i in range(4)] == list(zip(lower, upper))
    assert m.nn.x[1, 0].bounds == (np.tanh(lower[0]), np.tanh(upper[0]))


def solve_at(net: FeedForwardNet, x: np.ndarray, sense) -> float:
    """Optimize the output with the inputs fixed at x."""
    m = build(net)
    for i, value in enumerate(x):
        m.nn.x[0, i].fix(value)
    m.objective = pyo.Objective(expr=m.nn.y[0], sense=sense)
    result = pyo.SolverFactory("gurobi_direct").solve(m)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    return pyo.value(m.objective)


@pytest.mark.parametrize("widths", [[4], [3, 3]])
def test_relaxation_brackets_the_network(widths):
    net = random_net(widths, seed=1)
    for x in sample_points(net, 3):
        value = net.predict(x)[0, 0]
        assert solve_at(net, x, pyo.minimize) <= value + TOLERANCE
        assert solve_at(net, x, pyo.maximize) >= value - TOLERANCE


def grid_minimum(net: FeedForwardNet, n_per_input: int = 301) -> float:
    axes = np.linspace(net.lower_bound, net.upper_bound, n_per_input).T
    grid = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, net.n_inputs)
    return net.predict(grid).min()


@pytest.mark.parametrize(
    "net",
    [
        random_net([4, 3], seed=2),
        FeedForwardNet.load(ROOT / "outputs/tanh/models/peaks__tanh__h1x10__t0"),
    ],
    ids=["random_h2x4", "peaks__tanh__h1x10__t0"],
)
def test_bound_is_valid(net):
    solution = minimize_output(net, "tanh.pq", "gurobi_direct")
    assert np.all(net.lower_bound <= solution.x) and np.all(solution.x <= net.upper_bound)
    assert solution.bound <= solution.objective + TOLERANCE
    true_minimum = grid_minimum(net)
    assert solution.bound <= true_minimum + TOLERANCE
    assert true_minimum <= net.predict(solution.x)[0, 0] + TOLERANCE


def test_bound_tightens_with_the_tolerance():
    net = random_net([5, 4], seed=3)  # small, so that 40 pieces per neuron fit Gurobi's size limit
    bounds = [
        minimize_output(net, "tanh.pq", "gurobi_direct", tol=tol, max_pieces=40).bound
        for tol in (0.1, 0.02, 0.005)
    ]
    assert bounds[0] <= bounds[1] + TOLERANCE and bounds[1] <= bounds[2] + TOLERANCE
    assert bounds[2] <= grid_minimum(net) + TOLERANCE
