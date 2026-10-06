"""
ReLU embedding methods (Plate et al. 2026). All use the big-M encoding of eq. (5); each changes one of
the bounds, the network parameters, or the encoding.

Layers are k = 0 (inputs), 1, ..., K (hidden) and K + 1 (outputs), neurons i, as in the paper.
"""

import pyomo.environ as pyo

from hybopt.bounds import interval_bounds
from hybopt.net import FeedForwardNet


def add_bigm(block: pyo.Block, net: FeedForwardNet, bounds) -> None:
    """
    Add ``net`` to ``block`` with the big-M encoding (eq. 5), using the pre-activation ``bounds``
    (one (L, U) pair per layer, see ``hybopt.bounds``) as big-M coefficients.

    x[k, i]               : inputs (k = 0, within the box) and hidden-layer outputs x^(k)_i
                            (within [max(L, 0), max(U, 0)]).
    y[i]                  : outputs, y = W^(K+1) x^(K) + b^(K+1), within the output-layer bounds.
    pre_activation[k, i]  : the expression W^(k)_i x^(k-1) + b^(k)_i, for k = 1, ..., K + 1.
    z[k, i]               : binary, 1 if the neuron is active; only for unstable neurons (L < 0 < U),
                            which get the constraints above_pre_activation, below_if_active and
                            zero_if_inactive of eq. (5).

    Stable neurons need no binary (Section 2.1.2): stably_active sets x = pre-activation when L >= 0,
    and the bounds [0, 0] fix x to zero when U <= 0.
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
