from __future__ import annotations

import torch
import torch.distributions as dist
import beanmachine.ppl as bm
import copy
import random

from neuralpp.experiments.bm_integration.factor_dist import get_distribution
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from typing import NamedTuple, Collection, Optional


class DummyVar(NamedTuple):
    # BM assumes each RV is represented as a "Variable" struct inside of World and
    # most inference algorithms are going to interact with the "distribution" field.
    distribution: dist.Distribution


class FactorWorld(bm.world.World):
    """a "World" that explains factor graph to BM inference methods"""

    def __init__(self, factors, observations):
        self._factors = factors
        # for variables whose values are fixed during an inference
        self.observations = observations.copy()

        # collect variables that need to be inferred and the factor(s) they belong to
        self._var_to_factors = dict()
        for factor in factors:
            for var in factor.variables:
                if var in self._var_to_factors:
                    self._var_to_factors[var].append(factor)
                else:
                    # can't use set because factor is unhashable
                    self._var_to_factors[var] = [factor]

        self._variables = dict()
        for var in self._var_to_factors:
            self.initialize_value(var)

    def initialize_value(self, var: Variable) -> None:
        if var in self.observations:
            self._variables[var] = self.observations[var]
        elif isinstance(var, IntegerVariable):
            self._variables[var] = random.randint(0, var.cardinality - 1)
        else:
            # some dummy initialization to get this script running. This does not work
            # in general for distribution with limited support
            self._variables[var] = dist.Uniform(-2, 2).sample()

    def __getitem__(self, variable: Variable):
        return self._variables[variable]

    def replace(self, assignment_dict):
        # return a new world with updated values
        new_world = copy.copy(self)
        new_world._variables = {**self._variables, **assignment_dict}
        return new_world

    def log_prob(self, variables: Optional[Collection[Variable]] = None):
        """return the log prob of the factors conditioned on the current assignments
        of variables"""
        log_prob = 0.0

        if variables is None:
            factors = self._factors
        else:
            factors = set().union(*(self._var_to_factors[var] for var in variables))

        for factor in factors:
            # evaluate each factor on the assignments
            log_prob += factor(self._variables).log()
        return log_prob

    def get_variable(self, variable):
        return DummyVar(
            distribution=get_distribution(
                variable, self._var_to_factors[variable], self._variables
            )
        )
