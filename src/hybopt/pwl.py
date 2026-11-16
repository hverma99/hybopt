"""Single-neuron piecewise linear tanh constraints for Pyomo models."""

import math
from collections.abc import Iterable

import pyomo.environ as pyo
from pyomo.core.base.var import VarData


def piecewise_linear_tanh(
    x: VarData,
    y: VarData,
    breakpoints: Iterable[float],
    *,
    representation: str = "DCC",
) -> pyo.Piecewise:
    """Return a component enforcing y = PWL-tanh(x); attach it to your model.

    ``x`` must have finite bounds covered by strictly increasing breakpoints.
    Endpoint heights are exact tanh values; interpolation between them is an
    approximation. DCC uses binary variables; SOS2 needs solver SOS2 support.
    This function does not train a network or solve the model.

    Example::

        model.x = pyo.Var(bounds=(-2, 2))
        model.y = pyo.Var()
        model.tanh_pwl = piecewise_linear_tanh(
            model.x, model.y, [-2, -1, 0, 1, 2]
        )
    """
    if not isinstance(x, VarData) or not isinstance(y, VarData):
        raise TypeError("x and y must be scalar Pyomo variables or indexed entries")
    if representation not in ("DCC", "SOS2"):
        raise ValueError("representation must be 'DCC' or 'SOS2'")

    points = [float(point) for point in breakpoints]
    if len(points) < 2:
        raise ValueError("At least two breakpoints are required")
    if not all(math.isfinite(point) for point in points):
        raise ValueError("Breakpoints must be finite")
    if any(left >= right for left, right in zip(points, points[1:])):
        raise ValueError("Breakpoints must be strictly increasing")

    lower, upper = x.bounds
    if lower is None or upper is None or not all(
        math.isfinite(bound) for bound in (lower, upper)
    ):
        raise ValueError("x must have finite lower and upper bounds")
    if lower > upper:
        raise ValueError("x lower bound must not exceed its upper bound")
    if points[0] > lower or points[-1] < upper:
        raise ValueError("Breakpoints must cover the entire bounded range of x")

    return pyo.Piecewise(
        y,
        x,
        pw_pts=points,
        f_rule=[math.tanh(point) for point in points],
        pw_constr_type="EQ",
        pw_repn=representation,
    )
