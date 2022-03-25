import torch.distributions as dist
from torch.distributions import constraints
from torch.distributions import normal
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    ProductFactor,
)
from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
)
from neuralpp.inference.graphical_model.representation.factor.switch_factor import (
    SwitchFactor,
)
from neuralpp.inference.graphical_model.representation.factor.continuous.normal_factor import (
    NormalFactor,
)
from neuralpp.inference.graphical_model.representation.factor.table_factor import (
    TableFactor,
)
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.graphical_model.variable.variable import Variable
from typing import Dict, List, Union
import torch
from functools import reduce
import operator


class TableFactorDist(dist.Distribution):
    def __init__(self, variable: IntegerVariable, table_factor: PyTorchTableFactor):
        self.variable = variable
        self.factor = table_factor

    def sample(self):
        return self.factor.sample().squeeze()

    def log_prob(self, value):
        return self.factor({self.variable: value}).log()

    @property
    def support(self):
        return constraints.integer_interval(0, self.variable.cardinality)


class GenericFactorDist(dist.Distribution):
    def __init__(self, variable: Variable, factor: Factor):
        self.variable = variable
        self.factor = factor

    # Since NUTS does not invoke dist.log_prob or dist.sample, for the purpose of
    # running GMM example with NUTS/Compositional inference, it should be fine to
    # leave these two methods unimplemented.
    # we can figure out how to deal with them later
    def sample(self):
        # some dummy sampling to get pass initialization
        return dist.Uniform(-2, 2).sample()

    def log_prob(self, value):
        return self.factor({self.variable: value}).log()

    @property
    def support(self):
        # fall back to real given that there is no known constraints
        return constraints.real


def get_distribution(
    variable: Variable,
    factors: List[Factor],
    assignments: Dict[Variable, Union[torch.Tensor, int]],
):
    if isinstance(variable, IntegerVariable):
        table_factors = filter(lambda x: isinstance(x, PyTorchTableFactor), factors)
        # combine weights for multiple table factors (if needed)
        reduced = reduce(operator.mul, table_factors)
        return TableFactorDist(variable, reduced)
    else:
        return GenericFactorDist(variable, factors)


def get_distribution_v2(variable: Variable, factor: Factor):
    if isinstance(factor, TableFactor):
        return TableFactorDist(variable, factor)
    else:
        return GenericFactorDist(variable, factor)
