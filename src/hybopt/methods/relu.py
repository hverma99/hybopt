r"""
ReLU embedding methods (Plate et al. 2026). All use the big-M encoding of eq. (5); each changes one of
the bounds, the network parameters, or the encoding.

Layers are :math:`k = 0` (inputs), :math:`1, \dots, K` (hidden) and :math:`K + 1` (outputs), neurons
:math:`i`, as in the paper.
"""

import numpy as np
import pyomo.environ as pyo
from scipy.optimize import minimize

from hybopt.bounds import interval_bounds
from hybopt.net import FeedForwardNet


def add_bigm(block: pyo.Block, net: FeedForwardNet, bounds) -> None:
    r"""
    Add ``net`` to ``block`` with the big-M encoding (eq. 5), using the pre-activation ``bounds``
    (one (L, U) pair per layer, see ``hybopt.bounds``) as big-M coefficients.

    x[k, i]               : inputs (k = 0, within the box) and hidden-layer outputs :math:`x^{(k)}_i`
                            (within :math:`[\max(L, 0), \max(U, 0)]`).
    y[i]                  : outputs, :math:`y = W^{(K+1)} x^{(K)} + b^{(K+1)}`, within the output-layer
                            bounds.
    pre_activation[k, i]  : the expression :math:`W^{(k)}_i x^{(k-1)} + b^{(k)}_i`, for
                            :math:`k = 1, \dots, K + 1`.
    z[k, i]               : binary, 1 if the neuron is active; only for unstable neurons
                            (:math:`L < 0 < U`),
                            which get the constraints above_pre_activation, below_if_active and
                            zero_if_inactive of eq. (5).

    Stable neurons need no binary (Section 2.1.2): stably_active sets x = pre-activation when :math:`L \ge 0`,
    and the bounds :math:`[0, 0]` fix x to zero when :math:`U \le 0`.
    """
    sizes = net.layer_sizes
    n_hidden = len(sizes) - 2
    hidden = [(k, i) for k in range(1, n_hidden + 1) for i in range(sizes[k])]
    lower = {(k, i): float(bounds[k - 1][0][i]) for k, i in hidden}
    upper = {(k, i): float(bounds[k - 1][1][i]) for k, i in hidden}
    unstable = [n for n in hidden if lower[n] < 0 < upper[n]]
    stably_active = [n for n in hidden if lower[n] >= 0]

    def x_bounds(b, k, i):
        if k == 0:
            return float(net.lower_bound[i]), float(net.upper_bound[i])
        return max(lower[k, i], 0.0), max(upper[k, i], 0.0)

    def pre_activation(b, k, i):
        w, bias = net.weights[k - 1][i], net.biases[k - 1][i]
        return sum(w[j] * b.x[k - 1, j] for j in range(sizes[k - 1])) + bias

    output_lower, output_upper = bounds[-1]
    block.x = pyo.Var(
        [(k, i) for k in range(n_hidden + 1) for i in range(sizes[k])], bounds=x_bounds
    )
    block.y = pyo.Var(
        range(net.n_outputs),
        bounds=lambda b, i: (float(output_lower[i]), float(output_upper[i])),
    )
    block.pre_activation = pyo.Expression(
        [(k, i) for k in range(1, n_hidden + 2) for i in range(sizes[k])], rule=pre_activation
    )
    block.z = pyo.Var(unstable, within=pyo.Binary)

    block.above_pre_activation = pyo.Constraint(
        unstable, rule=lambda b, k, i: b.x[k, i] >= b.pre_activation[k, i]
    )
    block.below_if_active = pyo.Constraint(
        unstable,
        rule=lambda b, k, i: b.x[k, i]
        <= b.pre_activation[k, i] - lower[k, i] * (1 - b.z[k, i]),
    )
    block.zero_if_inactive = pyo.Constraint(
        unstable, rule=lambda b, k, i: b.x[k, i] <= upper[k, i] * b.z[k, i]
    )
    block.stably_active = pyo.Constraint(
        stably_active, rule=lambda b, k, i: b.x[k, i] == b.pre_activation[k, i]
    )
    block.output = pyo.Constraint(
        range(net.n_outputs), rule=lambda b, i: b.y[i] == b.pre_activation[n_hidden + 1, i]
    )


def bigm_ia(block: pyo.Block, net: FeedForwardNet) -> None:
    """
    R1 (relu.bigm_ia), the baseline: big-M encoding with interval-arithmetic bounds (eqs. 5-7).
    """
    add_bigm(block, net, interval_bounds(net))


# A-posteriori scaling (R3, Section 2.3). ReLU is positively homogeneous, relu(c a) = c relu(a) for
# c > 0, so scaling row i of layer k (weights and bias) by c^(k)_i and column i of layer k + 1 by
# 1 / c^(k)_i leaves the network's function unchanged (eqs. 9-10). Only hidden neurons are scaled;
# the inputs and outputs, and so the input box and the output bias, stay as they are.


def check_relu(net: FeedForwardNet, name: str) -> None:
    if net.activation != "relu":
        raise ValueError(f"{name} needs a relu network, got {net.activation}")


def log_factors_per_layer(net: FeedForwardNet, log_factors) -> list[np.ndarray]:
    r"""
    The log-scaling factors :math:`\tilde c^{(k)} = \log c^{(k)}` of every layer k = 0, ..., K + 1:
    zeros for the inputs and outputs and the given ones (a list with one array per hidden layer, or
    one flat array of all hidden neurons in layer order) in between.
    """
    hidden_sizes = net.layer_sizes[1:-1]
    if isinstance(log_factors, np.ndarray) and log_factors.ndim == 1:
        if len(log_factors) != sum(hidden_sizes):
            raise ValueError(
                f"log_factors needs {sum(hidden_sizes)} values, one per hidden neuron, got "
                f"{len(log_factors)}"
            )
        log_factors = np.split(log_factors, np.cumsum(hidden_sizes)[:-1])
    log_factors = [np.asarray(factors, dtype=float) for factors in log_factors]
    if [len(factors) for factors in log_factors] != hidden_sizes:
        raise ValueError(
            f"log_factors needs one array per hidden layer with sizes {hidden_sizes}, got sizes "
            f"{[len(factors) for factors in log_factors]}"
        )
    return [np.zeros(net.n_inputs), *log_factors, np.zeros(net.n_outputs)]


def rescale(net: FeedForwardNet, log_factors) -> FeedForwardNet:
    r"""
    The functionally identical network with hidden neuron i of layer k scaled by
    :math:`c^{(k)}_i = \exp(\tilde c^{(k)}_i)` (eq. 10):
    :math:`\tilde W^{(k)}_{ij} = W^{(k)}_{ij} c^{(k)}_i / c^{(k-1)}_j` and
    :math:`\tilde b^{(k)}_i = b^{(k)}_i c^{(k)}_i`, with :math:`c^{(0)} = c^{(K+1)} = 1`.

    log_factors : :math:`\tilde c`, one array per hidden layer or one flat array in layer order.
    """
    check_relu(net, "rescale")
    c = log_factors_per_layer(net, log_factors)
    weights = [
        w * np.exp(c[k][:, None] - c[k - 1][None, :]) for k, w in enumerate(net.weights, start=1)
    ]
    biases = [b * np.exp(c[k]) for k, b in enumerate(net.biases, start=1)]
    return FeedForwardNet(
        weights,
        biases,
        net.activation,
        net.lower_bound,
        net.upper_bound,
        net.input_names,
        net.output_names,
    )


def scaling_objective(net: FeedForwardNet, log_factors: np.ndarray) -> tuple[float, np.ndarray]:
    r"""
    The sum of the absolute weights and hidden biases of ``rescale(net, log_factors)`` (eq. 12) and
    its gradient with respect to the flat ``log_factors``.

    Each term is :math:`|W^{(k)}_{ij}| \exp(\tilde c^{(k)}_i - \tilde c^{(k-1)}_j)` or
    :math:`|b^{(k)}_i| \exp(\tilde c^{(k)}_i)`, so the problem is convex. It is written with
    :math:`|W|` rather than :math:`\exp(\log |W| + \dots)` so that zero weights need no special
    case. The last weight term uses the factors :math:`\tilde c^{(K)}_j` of the last hidden layer,
    as in eq. (10); eq. (12) prints :math:`\tilde c^{(J)}_i` there. The output bias is not scaled
    and is left out (eq. 11).
    """
    c = log_factors_per_layer(net, np.asarray(log_factors, dtype=float))
    n_hidden = len(c) - 2
    value = 0.0
    gradient = [np.zeros_like(factors) for factors in c]
    for k, (w, b) in enumerate(zip(net.weights, net.biases), start=1):
        terms = np.abs(w) * np.exp(c[k][:, None] - c[k - 1][None, :])
        value += terms.sum()
        gradient[k] += terms.sum(axis=1)
        gradient[k - 1] -= terms.sum(axis=0)
        if k <= n_hidden:
            bias_terms = np.abs(b) * np.exp(c[k])
            value += bias_terms.sum()
            gradient[k] += bias_terms
    return float(value), np.concatenate(gradient[1:-1])


def scale_relu(net: FeedForwardNet, max_log_factor: float = 50.0) -> FeedForwardNet:
    r"""
    The functionally identical network whose scaling factors minimize the absolute weights and
    biases (eq. 12). This changes the magnitudes of the big-M coefficients, not the tightness of the
    formulation (see ``bigm_scaled``).

    Solved with L-BFGS-B from :math:`\tilde c = 0` (the network as given), so the objective never
    increases. The bounds :math:`|\tilde c| \le` ``max_log_factor`` keep the factors finite where
    the infimum is not attained, e.g. for a neuron whose incoming weights and bias are all zero.
    """
    check_relu(net, "scale_relu")
    n_factors = sum(net.layer_sizes[1:-1])
    start = np.zeros(n_factors)
    result = minimize(
        lambda log_factors: scaling_objective(net, log_factors),
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-max_log_factor, max_log_factor)] * n_factors,
    )
    best = result.x if result.fun <= scaling_objective(net, start)[0] else start
    return rescale(net, best)


def bigm_scaled(block: pyo.Block, net: FeedForwardNet) -> None:
    """
    R3 (relu.bigm_scaled): R1 on the network rescaled by ``scale_relu`` (eqs. 9-12). The inputs
    x[0, i] and outputs y[i] are those of ``net``; the hidden variables x[k, i] hold the rescaled
    neurons.

    Interval arithmetic commutes with positive scaling: the bounds of a neuron scaled by c are c
    times its bounds, and the output bounds do not change. In exact arithmetic this model is
    therefore R1 after the change of variables x[k, i] -> c x[k, i], with the same binaries and the
    same LP relaxation, and no neuron becomes stable. Scaling changes the magnitude of the big-M
    coefficients, which matters to the solver through its absolute tolerances, not the tightness of
    the formulation.
    """
    bigm_ia(block, scale_relu(net))
