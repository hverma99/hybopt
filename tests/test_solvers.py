"""The solvers hybopt relies on are reachable through Pyomo under these names."""

import numpy as np
import pyomo.environ as pyo
import pytest

MILP_SOLVERS = ["appsi_highs", "gurobi_direct"]


def relu_milp() -> pyo.ConcreteModel:
    """
    min y + 0.5 x  s.t.  y = max(0, x), x in [-2, 3], as the big-M encoding of Plate et al. eq. (5)
    with L = -2, U = 3. The optimum is x = -2, y = 0 (inactive neuron), objective -1.
    """
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-2, 3))
    m.y = pyo.Var(within=pyo.NonNegativeReals)
    m.z = pyo.Var(within=pyo.Binary)
    m.above_input = pyo.Constraint(expr=m.y >= m.x)
    m.input_if_active = pyo.Constraint(expr=m.y <= m.x + 2 * (1 - m.z))
    m.zero_if_inactive = pyo.Constraint(expr=m.y <= 3 * m.z)
    m.objective = pyo.Objective(expr=m.y + 0.5 * m.x)
    return m


@pytest.mark.parametrize("solver", MILP_SOLVERS)
def test_milp_solver(solver):
    m = relu_milp()
    result = pyo.SolverFactory(solver).solve(m)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert pyo.value(m.objective) == pytest.approx(-1.0)
    assert pyo.value(m.z) == pytest.approx(0.0)


def test_gurobi_solves_tanh_written_with_exp():
    # min (t - 0.5)^2 with t = tanh(x) in form F3 = 1 - 2 / (exp(2x) + 1): x = atanh(0.5)
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(-3, 3))
    m.t = pyo.Var(bounds=(-1, 1))
    m.tanh = pyo.Constraint(expr=m.t == 1 - 2 / (pyo.exp(2 * m.x) + 1))
    m.objective = pyo.Objective(expr=(m.t - 0.5) ** 2)
    result = pyo.SolverFactory("gurobi_direct_minlp").solve(m)
    assert result.solver.termination_condition == pyo.TerminationCondition.optimal
    assert pyo.value(m.x) == pytest.approx(np.arctanh(0.5), abs=1e-4)
