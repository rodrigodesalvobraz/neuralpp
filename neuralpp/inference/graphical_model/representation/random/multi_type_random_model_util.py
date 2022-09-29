from random import random
from typing import List

import torch

from neuralpp.inference.graphical_model.representation.factor.continuous.normal_factor import (
    NormalFactor,
)
from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
)
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.factor.switch_factor import (
    SwitchFactor,
)
from neuralpp.inference.graphical_model.variable.integer_variable import (
    IntegerVariable,
)
from neuralpp.inference.graphical_model.variable.tensor_variable import (
    TensorVariable,
)
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util.util import is_iterable, repeat


def make_randomly_shifted_standard_gaussian_given_range(variables, range):
    mean_shift = torch.rand(tuple()) * range
    return make_shifted_standard_gaussian_given_shift(variables, mean_shift)


def make_shifted_standard_gaussian_given_shift(variables, mean_shift):
    if is_iterable(variables):
        variable = variables[0]
    else:
        variable = variables
    mu = TensorVariable(f"mu_{{{variable}}}")
    std_dev = TensorVariable(f"std_dev_{{{variable}}}")
    return NormalFactor(
        [variable, mu, std_dev],
        conditioning_dict={mu: mean_shift, std_dev: torch.tensor(1.0)},
    )


def make_standard_gaussian(variables):
    if is_iterable(variables):
        variable = variables[0]
    else:
        variable = variables
    mu = TensorVariable(f"mu_{{{variable}}}")
    std_dev = TensorVariable(f"std_dev_{{{variable}}}")
    return NormalFactor(
        [variable, mu, std_dev],
        conditioning_dict={mu: torch.tensor(0.0), std_dev: torch.tensor(1.0)},
    )


def make_gaussian_with_mean(*args):
    variable, mu = args[0] if len(args) == 1 else args
    std_dev = TensorVariable(f"std_dev_{{{variable}}}")
    return NormalFactor(
        [variable, mu, std_dev],
        conditioning_dict={std_dev: torch.tensor(1.0)},
    )


def make_switch_of_gaussians_with_mean(*args):
    true_arguments = args[0] if len(args) == 1 else args
    variable = true_arguments[0]
    switch = true_arguments[1]
    mus = true_arguments[2:]
    assert switch.cardinality == len(mus)
    components = [make_gaussian_with_mean([variable, mu]) for mu in mus]
    return SwitchFactor(switch, components)


def random_categorical_probabilities(k: int) -> List[float]:
    """
    Returns a list with the probabilities of a categorical distribution of dimension k.
    """
    unnormalized = repeat(k, lambda: random())
    normalizing_constant = sum(unnormalized)
    normalized = [u / normalizing_constant for u in unnormalized]
    return normalized


def random_categorical_probabilities_table(dimensions) -> List:
    """
    Returns a table of given dimensions filled with probabilities,
    where the last dimension is normalized (sums up to 1).
    """
    assert len(dimensions) > 0
    if len(dimensions) == 1:
        return random_categorical_probabilities(dimensions[0])
    else:
        return repeat(
            dimensions[0],
            lambda: random_categorical_probabilities_table(dimensions[1:]),
        )


def make_random_table_factor(variables: List[IntegerVariable]) -> Factor:
    dimensions = [v.cardinality for v in variables]
    return PyTorchTableFactor(
        variables, random_categorical_probabilities_table(dimensions)
    )
