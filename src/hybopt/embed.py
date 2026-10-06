"""
Embed a trained network in a Pyomo model as one block, with the method chosen by registry key:

    m.nn = embed(net, "relu.bigm_ia", inputs=m.x, outputs=m.y)  # tied to the model's variables
    m.nn = embed(net, "relu.bigm_ia")  # use the block's own m.nn.x[0, i] and m.nn.y[i]

Several networks can be embedded in one model, each in its own block.
"""

import pyomo.environ as pyo

from hybopt.methods import get_method
from hybopt.net import FeedForwardNet


def variable_list(variables, n_variables: int, label: str) -> list:
    """
    ``variables`` (an indexed Var, a single Var or a list of variables) as a list of n_variables.
    """
    if isinstance(variables, pyo.Var):
        variables = list(variables.values()) if variables.is_indexed() else [variables]
    variables = list(variables)
    if len(variables) != n_variables:
        raise ValueError(
            f"{label} needs {n_variables} variables, one per network {label[:-1]}, "
            f"got {len(variables)}"
        )
    return variables


def embed(net: FeedForwardNet, method: str, inputs=None, outputs=None) -> pyo.Block:
    """
    A Pyomo block holding ``net`` encoded with ``method`` (a key of ``hybopt.methods.METHODS``);
    attach it to a model with ``m.nn = embed(...)``.

    inputs, outputs : The model's variables to tie to the network inputs and outputs, in the order of
                      net.input_names and net.output_names: an indexed Var, a single Var or a list.
                      Leave out to work with the block's own variables x[0, i] and y[i].

    The block keeps its inputs within the network's box, where the encoding is valid.
    """
    build = get_method(method, net.activation)
    if inputs is not None:
        inputs = variable_list(inputs, net.n_inputs, "inputs")
    if outputs is not None:
        outputs = variable_list(outputs, net.n_outputs, "outputs")

    block = pyo.Block(concrete=True)
    build(block, net)
    if inputs is not None:
        block.link_inputs = pyo.Constraint(
            range(net.n_inputs), rule=lambda b, i: inputs[i] == b.x[0, i]
        )
    if outputs is not None:
        block.link_outputs = pyo.Constraint(
            range(net.n_outputs), rule=lambda b, i: outputs[i] == b.y[i]
        )
    return block
