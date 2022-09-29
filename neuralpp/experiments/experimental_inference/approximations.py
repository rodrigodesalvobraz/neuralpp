from neuralpp.experiments.experimental_inference.graph_analysis import (
    FactorTree,
    FactorGraph,
)
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)

"""
Each approximation scheme is represented by a function responsible for producing its trivial approximations.
It takes three parameters: node (Variable or Factor), partial_tree: FactorTree, and full_tree: FactorTree.
"""


def message_approximation(node, partial_tree: FactorTree, full_tree: FactorTree):
    return PyTorchTableFactor.from_function(
        FactorGraph.variables_at(node), lambda *args: 1.0
    )
