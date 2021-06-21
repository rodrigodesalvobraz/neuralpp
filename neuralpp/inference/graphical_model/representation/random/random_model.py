import random

import torch

from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import PyTorchTableFactor
from neuralpp.inference.graphical_model.representation.table.table_util import shape
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.util.util import join


def generate_model(number_of_factors, number_of_variables, cardinality, factor_class=PyTorchTableFactor):
    """
    Makes a random model according to parameters, constructing factors using factor_class' from_function method.
    """
    variables = {IntegerVariable(f"x{i}", cardinality) for i in range(number_of_variables)}
    factors = [generate_factor(random_subset(variables), factor_class) for i1 in range(number_of_factors)]
    missing_variables = variables - {v for f in factors for v in f.variables}
    uniform_factors_for_missing_variables = [factor_class.from_function([m], lambda vm: 1.0) for m in missing_variables]
    return factors + uniform_factors_for_missing_variables


def generate_factor(variables, factor_class):
    return factor_class(variables, torch.rand(shape(variables), requires_grad=True))


def random_subset(variables):
    result = []
    for variable in variables:
        if bool(random.getrandbits(1)):
            result.append(variable)
    return result


if __name__ == "__main__":
    print(join(generate_model(number_of_factors=4, number_of_variables=5, cardinality=3), "\n"))
