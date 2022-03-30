from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
)
from neuralpp.inference.graphical_model.variable.variable import Variable
import beanmachine.ppl as bm
from typing import Callable, List, overload, Union
from neuralpp.experiments.bm_integration.factor_dist import get_distribution_v2
import torch
from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    ProductFactor,
)


# a mapping from variable to the corresponding function
rv_functions = {}


def get_value(variable: Variable) -> torch.Tensor:
    return rv_functions[variable]()


def make_random_variable(factor: Factor) -> None:
    if isinstance(factor, ProductFactor):
        for f in ProductFactor.factors(factor):
            make_random_variable(f)
        return

    parent_vars = factor.variables[1:]
    child_var = factor.variables[0]

    @bm.random_variable
    def rvfunction():
        parent_values = {p: get_value(p) for p in parent_vars}
        factor_on_child = factor.condition(parent_values)
        return get_distribution_v2(child_var, factor_on_child)

    rv_functions[child_var] = rvfunction


def make_functional(variable: Variable, value: torch.Tensor):
    @bm.functional
    def rvfunction():
        return value

    rv_functions[variable] = rvfunction
