from neuralpp.experiments.experimental_inference.graph_analysis import FactorTree
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import PyTorchTableFactor
from neuralpp.inference.graphical_model.variable.variable import Variable


def uniform_approximation_fn(node, partial_tree: FactorTree, full_tree: FactorTree):
    variables = [node] if isinstance(node, Variable) else node.variables
    # TODO: probably good to implement this with a UniformFactor which doesn't store an actual table
    return PyTorchTableFactor.from_function(variables, lambda *args: 1.0)
