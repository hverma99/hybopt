import pyomo.environ as pyo
import pytest

from hybopt.embed import embed
from hybopt.net import FeedForwardNet

# y = 2 relu(x - 1) - relu(x) on [-2, 3]: minimum -1 at x = 1
KINK = FeedForwardNet([[[1.0], [1.0]], [[2.0, -1.0]]], [[-1.0, 0.0], [0.0]], "relu", [-2.0], [3.0])
# y = |x1 - x2| on [-5, 5]^2
ABS_DIFF = FeedForwardNet(
    [[[1.0, -1.0], [-1.0, 1.0]], [[1.0, 1.0]]], [[0.0, 0.0], [0.0]], "relu", [-5.0, -5.0], [5.0, 5.0]
)


def solve(m: pyo.ConcreteModel) -> None:
    result = pyo.SolverFactory("appsi_highs").solve(m)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal


def test_ties_the_model_variables():
    m = pyo.ConcreteModel()
    m.u = pyo.Var()
    m.v = pyo.Var()
    m.nn = embed(KINK, "relu.bigm_ia", inputs=m.u, outputs=m.v)
    m.at_least_two = pyo.Constraint(expr=m.u >= 2)
    m.objective = pyo.Objective(expr=m.v)
    solve(m)
    assert pyo.value(m.u) == pytest.approx(2.0)
    assert pyo.value(m.v) == pytest.approx(0.0, abs=1e-9)


def test_ties_indexed_variables_in_order():
    m = pyo.ConcreteModel()
    m.u = pyo.Var(["a", "b"])
    m.v = pyo.Var([0])
    m.nn = embed(ABS_DIFF, "relu.bigm_ia", inputs=m.u, outputs=m.v)
    m.u["a"].fix(3.0)
    m.u["b"].fix(-1.0)
    m.objective = pyo.Objective(expr=m.v[0])
    solve(m)
    assert pyo.value(m.nn.x[0, 0]) == pytest.approx(3.0)
    assert pyo.value(m.v[0]) == pytest.approx(4.0)


@pytest.mark.parametrize("sense, expected", [(pyo.minimize, -2.0), (pyo.maximize, 3.0)])
def test_keeps_inputs_in_the_box(sense, expected):
    m = pyo.ConcreteModel()
    m.u = pyo.Var()
    m.nn = embed(KINK, "relu.bigm_ia", inputs=[m.u])
    m.objective = pyo.Objective(expr=m.u, sense=sense)
    solve(m)
    assert pyo.value(m.u) == pytest.approx(expected)


def test_several_networks_in_one_model():
    half_relu = FeedForwardNet([[[1.0]], [[0.5]]], [[0.0], [0.0]], "relu", [-2.0], [3.0])
    m = pyo.ConcreteModel()
    m.u = pyo.Var()
    m.kink = embed(KINK, "relu.bigm_ia", inputs=[m.u])
    m.half = embed(half_relu, "relu.bigm_ia", inputs=[m.u])
    # 2 relu(u - 1) - relu(u) + 0.5 relu(u): minimum -0.5 at u = 1
    m.objective = pyo.Objective(expr=m.kink.y[0] + m.half.y[0])
    solve(m)
    assert pyo.value(m.u) == pytest.approx(1.0)
    assert pyo.value(m.objective) == pytest.approx(-0.5)


def test_without_model_variables():
    block = embed(KINK, "relu.bigm_ia")
    assert block.x[0, 0].bounds == (-2.0, 3.0)
    assert len(block.y) == 1
    assert not hasattr(block, "link_inputs") and not hasattr(block, "link_outputs")


def test_rejects_wrong_number_of_variables():
    m = pyo.ConcreteModel()
    m.u = pyo.Var([0, 1])
    with pytest.raises(ValueError, match="inputs needs 1 variables, one per network input, got 2"):
        embed(KINK, "relu.bigm_ia", inputs=m.u)
    with pytest.raises(ValueError, match="outputs needs 1 variables, one per network output, got 2"):
        embed(KINK, "relu.bigm_ia", outputs=[m.u[0], m.u[1]])


def test_rejects_unknown_method_and_wrong_activation():
    tanh_net = FeedForwardNet([[[1.0]], [[1.0]]], [[0.0], [0.0]], "tanh", [-1.0], [1.0])
    with pytest.raises(KeyError, match="unknown method 'relu.nope'"):
        embed(KINK, "relu.nope")
    with pytest.raises(ValueError, match="is for relu networks, but the network uses tanh"):
        embed(tanh_net, "relu.bigm_ia")
