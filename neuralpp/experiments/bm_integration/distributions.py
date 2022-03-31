import torch.distributions as dist
from torch.distributions import constraints
from torch.distributions import normal
from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
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
    def __init__(self, variable: IntegerVariable, table_factor: TableFactor):
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


def get_distribution(variable: Variable, factor: Factor):
    if isinstance(factor, TableFactor):
        assert isinstance(variable, IntegerVariable)
        return TableFactorDist(variable, factor)
    else:
        return GenericFactorDist(variable, factor)
