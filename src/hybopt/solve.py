"""
Minimize a network output over the network's input box, with the network embedded by a chosen method:

    solution = minimize_output(net, "relu.bigm_ia", solver="gurobi_direct")
    solution.x, solution.objective
"""

from dataclasses import dataclass

import numpy as np
import pyomo.environ as pyo

from hybopt.embed import embed
from hybopt.net import FeedForwardNet


@dataclass
class Solution:
    """
    An optimal solution of ``minimize_output``.

    x         : Optimal inputs, shape (n_inputs,).
    objective : The minimized network output at x.
    model     : The solved Pyomo model; the network is in model.nn.
    results   : Pyomo's solver results (status, bounds, ...).
    """

    x: np.ndarray
    objective: float
    model: pyo.ConcreteModel
    results: object


def output_index(net: FeedForwardNet, output: str | None) -> int:
    """
    Position of the output named ``output``; may be left out for a network with one output.
    """
    if output is None:
        if net.n_outputs != 1:
            raise ValueError(
                f"the network has {net.n_outputs} outputs {net.output_names}; choose one with output"
            )
        return 0
    if output not in net.output_names:
        raise ValueError(f"output must be one of {net.output_names}, got {output!r}")
    return net.output_names.index(output)


def minimize_output(
    net: FeedForwardNet, method: str, solver: str = "appsi_highs", output: str | None = None
) -> Solution:
    """
    Minimize one output of ``net`` over its input box, with the network embedded by ``method``.

    solver : Pyomo solver name, e.g. "appsi_highs" or "gurobi_direct" for the ReLU methods.
    output : Name of the output to minimize; may be left out for a network with one output.
    """
    index = output_index(net, output)
    optimizer = pyo.SolverFactory(solver)
    if not optimizer.available(exception_flag=False):
        raise ValueError(f"solver {solver!r} is not available to Pyomo")

    model = pyo.ConcreteModel()
    model.nn = embed(net, method)
    model.objective = pyo.Objective(expr=model.nn.y[index])
    results = optimizer.solve(model)

    termination = results.solver.termination_condition
    if termination != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"{solver} stopped with {termination}, not an optimal solution")
    return Solution(
        x=np.array([pyo.value(model.nn.x[0, i]) for i in range(net.n_inputs)]),
        objective=pyo.value(model.objective),
        model=model,
        results=results,
    )
