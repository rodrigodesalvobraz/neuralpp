from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
)
from neuralpp.inference.graphical_model.variable.variable import Variable
import beanmachine.ppl as bm
from typing import Callable
from neuralpp.experiments.bm_integration.factor_dist import get_distribution_v2
import torch


# a mapping from variable to the corresponding function
rv_functions = {}


def get_value(variable: Variable) -> torch.Tensor:
    return rv_functions[variable]()


def make_random_variable(factor: Factor) -> Callable:
    parent_vars = factor.variables[1:]
    child_var = factor.variables[0]

    @bm.random_variable
    def rvfunction():
        parent_values = {p: get_value(p) for p in parent_vars}
        factor_on_child = factor.condition(parent_values)
        return get_distribution_v2(child_var, factor_on_child)

    rv_functions[child_var] = rvfunction
    return rvfunction


def make_functional(variable: Variable, value: torch.Tensor) -> Callable:
    @bm.functional
    def rvfunction():
        return value

    rv_functions[variable] = rvfunction
    return rvfunction
