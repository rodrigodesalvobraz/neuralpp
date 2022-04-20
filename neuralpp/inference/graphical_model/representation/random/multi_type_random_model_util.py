import torch

from neuralpp.inference.graphical_model.representation.factor.continuous.normal_factor import NormalFactor
from neuralpp.inference.graphical_model.representation.factor.switch_factor import SwitchFactor
from neuralpp.inference.graphical_model.variable.tensor_variable import TensorVariable
from neuralpp.util.util import is_iterable


def make_randomly_shifted_standard_gaussian_given_range(variables, range):
    mean_shift = torch.rand(tuple())*range
    return make_shifted_standard_gaussian_given_shift(variables, mean_shift)


def make_shifted_standard_gaussian_given_shift(variables, mean_shift):
    if is_iterable(variables):
        variable = variables[0]
    else:
        variable = variables
    mu = TensorVariable(f"mu_{{{variable}}}")
    std_dev = TensorVariable(f"std_dev_{{{variable}}}")
    return NormalFactor([variable, mu, std_dev], conditioning_dict={mu: mean_shift, std_dev: torch.tensor(1.0)})


def make_standard_gaussian(variables):
    if is_iterable(variables):
        variable = variables[0]
    else:
        variable = variables
    mu = TensorVariable(f"mu_{{{variable}}}")
    std_dev = TensorVariable(f"std_dev_{{{variable}}}")
    return NormalFactor([variable, mu, std_dev], conditioning_dict={mu: torch.tensor(0.0), std_dev: torch.tensor(1.0)})


def make_gaussian_with_mean(*args):
    variable, mu = args[0] if len(args) == 1 else args
    std_dev = TensorVariable(f"std_dev_{{{variable}}}")
    return NormalFactor([variable, mu, std_dev], conditioning_dict={std_dev: torch.tensor(1.0)})


def make_switch_of_gaussians_with_mean(*args):
    variable, switch, mu1, mu2 = args[0] if len(args) == 1 else args
    assert switch.cardinality == 2
    return SwitchFactor(
        switch,
        [make_gaussian_with_mean([variable, mu1]), make_gaussian_with_mean([variable, mu2])])


