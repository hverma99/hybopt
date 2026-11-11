import math

import pyomo.environ as pyo
import pytest

from hybopt.pwl import piecewise_linear_tanh


def model_with_bounds(bounds=(-2, 2)):
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=bounds)
    model.y = pyo.Var()
    return model


@pytest.mark.parametrize("points", [[0], [-2, 0, 0, 2], [2, 0, -2], [-2, math.nan, 2]])
def test_rejects_invalid_breakpoints(points):
    model = model_with_bounds()
    with pytest.raises(ValueError):
        piecewise_linear_tanh(model.x, model.y, points)


@pytest.mark.parametrize("bounds", [(None, 2), (-2, None), (-3, 2)])
def test_requires_bounds_covered_by_breakpoints(bounds):
    model = model_with_bounds(bounds)
    with pytest.raises(ValueError):
        piecewise_linear_tanh(model.x, model.y, [-2, 0, 2])


@pytest.mark.parametrize("representation", ["DCC", "SOS2"])
def test_builds_supported_representations(representation):
    model = model_with_bounds()
    model.pwl = piecewise_linear_tanh(
        model.x, model.y, [-2, -1, 0, 1, 2], representation=representation
    )
    assert list(model.pwl.component_data_objects(pyo.Constraint, active=True))


@pytest.mark.parametrize("x", [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
@pytest.mark.parametrize("sense", [pyo.minimize, pyo.maximize])
def test_solver_enforces_interpolation(x, sense):
    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        pytest.skip("HiGHS is not installed")
    model = model_with_bounds()
    model.pwl = piecewise_linear_tanh(model.x, model.y, [-2, -1, 0, 1, 2])
    model.x.fix(x)
    model.objective = pyo.Objective(expr=model.y, sense=sense)
    results = solver.solve(model)
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    left = min(math.floor(x), 1)
    expected = math.tanh(left) + (x - left) * (math.tanh(left + 1) - math.tanh(left))
    assert pyo.value(model.y) == pytest.approx(expected, abs=1e-8)
