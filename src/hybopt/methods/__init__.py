"""
Embedding methods by registry key; the keys are used across code, tests and benchmark tables.

A method adds a network to a Pyomo block, ``method(block, net)``, with input variables x[0, i] within
the network's box and output variables y[i], which ``hybopt.embed`` ties to the model. Each key starts
with the activation the method is for ("relu." or "tanh."). Keyword options of a method (e.g. tol for
"tanh.pq") are passed through ``embed`` and ``minimize_output``.
"""

from hybopt.methods.relu import bigm_ia, bigm_scaled
from hybopt.methods.tanh import pq

METHODS = {"relu.bigm_ia": bigm_ia, "relu.bigm_scaled": bigm_scaled, "tanh.pq": pq}


def get_method(key: str, activation: str):
    """
    The method registered as ``key``, after checking that it is made for ``activation`` networks.
    """
    if key not in METHODS:
        raise KeyError(f"unknown method {key!r}; available: {sorted(METHODS)}")
    method_activation = key.split(".")[0]
    if method_activation != activation:
        raise ValueError(
            f"method {key!r} is for {method_activation} networks, but the network uses {activation}"
        )
    return METHODS[key]
