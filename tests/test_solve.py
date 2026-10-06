from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import pytest

from hybopt.net import FeedForwardNet
from hybopt.solve import minimize_output

ROOT = Path(__file__).resolve().parents[1]
SOLVERS = ["appsi_highs", "gurobi_direct"]
TOLERANCE = 1e-5  # MILP feasibility and integrality tolerances allow small deviations

# y = 2 relu(x - 1) - relu(x) on [-2, 3]: minimum -1 at x = 1
KINK = FeedForwardNet([[[1.0], [1.0]], [[2.0, -1.0]]], [[-1.0, 0.0], [0.0]], "relu", [-2.0], [3.0])
# y1 = relu(x) and y2 = -relu(x) on [-2, 3]
TWO_OUTPUTS = FeedForwardNet(
    [[[1.0]], [[1.0], [-1.0]]], [[0.0], [0.0, 0.0]], "relu", [-2.0], [3.0], output_names=["y1", "y2"]
)


def grid_minimum(net: FeedForwardNet, n_per_input: int = 301) -> float:
    axes = np.linspace(net.lower_bound, net.upper_bound, n_per_input).T
    grid = np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, net.n_inputs)
    return net.predict(grid).min()


@pytest.mark.parametrize("solver", SOLVERS)
def test_known_optimum(solver):
    solution = minimize_output(KINK, "relu.bigm_ia", solver)
    assert solution.x == pytest.approx([1.0], abs=TOLERANCE)
    assert solution.objective == pytest.approx(-1.0, abs=TOLERANCE)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "name", ["peaks__relu__h1x10__t0", "peaks__relu__h2x10__t0", "ackley__relu__h1x10__t0"]
)
def test_global_optimum_of_saved_networks(name, solver):
    net = FeedForwardNet.load(ROOT / "outputs/relu/models" / name)
    solution = minimize_output(net, "relu.bigm_ia", solver)
    assert np.all(net.lower_bound <= solution.x) and np.all(solution.x <= net.upper_bound)
    assert solution.objective == pytest.approx(net.predict(solution.x)[0, 0], abs=TOLERANCE)
    assert solution.objective <= grid_minimum(net) + TOLERANCE  # no grid point is better


def test_solvers_agree():
    net = FeedForwardNet.load(ROOT / "outputs/relu/models/peaks__relu__h2x10__t0")
    objectives = [minimize_output(net, "relu.bigm_ia", solver).objective for solver in SOLVERS]
    assert objectives[0] == pytest.approx(objectives[1], abs=TOLERANCE)


def test_solution_holds_the_solved_model():
    solution = minimize_output(KINK, "relu.bigm_ia")
    assert solution.results.solver.termination_condition == pyo.TerminationCondition.optimal
    assert pyo.value(solution.model.nn.x[0, 0]) == pytest.approx(solution.x[0])
    assert pyo.value(solution.model.objective) == solution.objective


@pytest.mark.parametrize("output, x, objective", [("y1", None, 0.0), ("y2", 3.0, -3.0)])
def test_minimizes_the_named_output(output, x, objective):
    solution = minimize_output(TWO_OUTPUTS, "relu.bigm_ia", output=output)
    assert solution.objective == pytest.approx(objective, abs=TOLERANCE)
    if x is not None:
        assert solution.x == pytest.approx([x], abs=TOLERANCE)


def test_output_must_be_chosen_for_several_outputs():
    with pytest.raises(ValueError, match=r"the network has 2 outputs \['y1', 'y2'\]"):
        minimize_output(TWO_OUTPUTS, "relu.bigm_ia")


def test_unknown_output():
    with pytest.raises(ValueError, match=r"output must be one of \['y1', 'y2'\], got 'y3'"):
        minimize_output(TWO_OUTPUTS, "relu.bigm_ia", output="y3")


def test_unavailable_solver():
    with pytest.raises(ValueError, match="solver 'no_such_solver' is not available"):
        minimize_output(KINK, "relu.bigm_ia", solver="no_such_solver")
