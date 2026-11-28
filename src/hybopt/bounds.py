r"""
Bounds :math:`L^{(k)} \le W^{(k)} x^{(k-1)} + b^{(k)} \le U^{(k)}` on the pre-activations of every
layer :math:`k`, valid for all inputs in the box. For ReLU networks they are the big-M coefficients of Plate et al. (2026), eq. (5).

Bounds are a list with one (L, U) pair of arrays per layer: bounds[k - 1] belongs to layer k, from the
first hidden layer to the output layer.
"""

import numpy as np

from hybopt.net import ACTIVATION_FUNCTIONS, FeedForwardNet


def interval_bounds(net: FeedForwardNet) -> list[tuple[np.ndarray, np.ndarray]]:
    r"""
    Pre-activation bounds of every layer by interval arithmetic (Plate et al. 2026, eqs. 6-7).

    Each layer takes the bounds of its inputs :math:`x^{(k-1)}`: the box for :math:`k = 1`, else the
    previous layer's bounds passed through the activation,
    :math:`\sigma(L) \le x^{(k-1)} \le \sigma(U)`, which holds for any nondecreasing activation
    (ReLU and tanh).
    """
    sigma = ACTIVATION_FUNCTIONS[net.activation]
    x_lower, x_upper = net.lower_bound, net.upper_bound
    bounds = []
    for w, b in zip(net.weights, net.biases):
        w_positive, w_negative = np.maximum(w, 0.0), np.minimum(w, 0.0)
        lower = w_positive @ x_lower + w_negative @ x_upper + b
        upper = w_positive @ x_upper + w_negative @ x_lower + b
        bounds.append((lower, upper))
        x_lower, x_upper = sigma(lower), sigma(upper)
    return bounds


def obbt_bounds(
    net: FeedForwardNet, solver: str = "gurobi_persistent", tolerance: float = 1e-6
) -> list[tuple[np.ndarray, np.ndarray]]:
    r"""
    Pre-activation bounds of a ReLU network by LP-based bound tightening (Plate et al. 2026, eq. 8):
    the bounds of each neuron are the minimum and maximum of its pre-activation over the LP relaxation
    of the big-M encoding (eq. 5, binaries relaxed to [0, 1]) of the layers before it.

    The layers are tightened in forward order, each with the bounds already tightened for the layers
    before it. Layer 1 keeps its interval bounds, which are exact for an affine function of the box.
    Every bound is widened by ``tolerance`` against the LP solver's feasibility tolerance and never
    looser than the interval bound.

    solver    : Pyomo name of an LP solver. With a persistent one (default "gurobi_persistent") the
                model of each layer is sent to the solver once and only the objective changes.
    tolerance : Absolute margin added outward to every LP bound.
    """
    # imported here because hybopt.methods.relu imports this module
    import pyomo.environ as pyo
    from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

    from hybopt.methods.relu import add_bigm

    if net.activation != "relu":
        raise ValueError(f"obbt_bounds needs a relu network, got {net.activation}")
    optimizer = pyo.SolverFactory(solver)
    if not optimizer.available(exception_flag=False):
        raise ValueError(f"solver {solver!r} is not available to Pyomo")
    persistent = isinstance(optimizer, PersistentSolver)

    bounds = interval_bounds(net)
    for k in range(2, len(net.weights) + 1):
        # the network up to layer k, whose outputs y[i] are the pre-activations of layer k
        truncated = FeedForwardNet(
            net.weights[:k], net.biases[:k], "relu", net.lower_bound, net.upper_bound
        )
        m = pyo.ConcreteModel()
        m.nn = pyo.Block()
        add_bigm(m.nn, truncated, bounds[:k])
        m.nn.z.domain = pyo.UnitInterval
        for y in m.nn.y.values():
            y.setlb(None)
            y.setub(None)
        m.objective = pyo.Objective(expr=m.nn.y[0])
        if persistent:
            optimizer.set_instance(m)

        lower, upper = (array.copy() for array in bounds[k - 1])
        for i in range(len(lower)):
            for sense in (pyo.minimize, pyo.maximize):
                m.objective.expr, m.objective.sense = m.nn.y[i], sense
                if persistent:
                    optimizer.set_objective(m.objective)
                    results = optimizer.solve()
                else:
                    results = optimizer.solve(m)
                termination = results.solver.termination_condition
                if termination != pyo.TerminationCondition.optimal:
                    raise RuntimeError(
                        f"{solver} stopped with {termination} on the bound LP of layer {k}, "
                        f"neuron {i}"
                    )
                value = pyo.value(m.objective)
                if sense == pyo.minimize:
                    lower[i] = max(lower[i], value - tolerance)
                else:
                    upper[i] = min(upper[i], value + tolerance)
        bounds[k - 1] = (lower, upper)
    return bounds
