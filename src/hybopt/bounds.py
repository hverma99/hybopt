"""
Bounds L^(k) <= W^(k) x^(k-1) + b^(k) <= U^(k) on the pre-activations of every layer k, valid for all
inputs in the box. For ReLU networks they are the big-M coefficients of Plate et al. (2026), eq. (5).

Bounds are a list with one (L, U) pair of arrays per layer: bounds[k - 1] belongs to layer k, from the
first hidden layer to the output layer.
"""

import numpy as np

from hybopt.net import ACTIVATION_FUNCTIONS, FeedForwardNet


def interval_bounds(net: FeedForwardNet) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Pre-activation bounds of every layer by interval arithmetic (Plate et al. 2026, eqs. 6-7).

    Each layer takes the bounds of its inputs x^(k-1): the box for k = 1, else the previous layer's
    bounds passed through the activation, sigma(L) <= x^(k-1) <= sigma(U), which holds for any
    nondecreasing activation (ReLU and tanh).
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
