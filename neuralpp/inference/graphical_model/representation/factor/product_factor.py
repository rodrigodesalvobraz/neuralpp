import itertools
import math
from typing import List

from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
)
from neuralpp.util import util
from neuralpp.util.group import Group
from neuralpp.util.util import join, split


class ProductFactor(Factor):
    def __init__(self, factors: List[Factor]):
        # collect all variables
        factors = [f for f in factors if f is not Group.identity]
        variables = itertools.chain.from_iterable(f.variables for f in factors)
        # getting unique variables while keeping the order (dict maintain insertion
        # order by default)
        variables = dict.fromkeys(variables).keys()
        super().__init__(list(variables))
        self._factors = util.flatten_one_level(
            factors,
            util.isinstance_predicate(ProductFactor),
            ProductFactor.factors,
        )

    def call_after_validation(self, assignment_dict, assignment_values):
        return math.prod(f(assignment_dict) for f in self._factors)

    def condition_on_non_empty_dict(self, assignment_dict):
        return ProductFactor(list(f.condition(assignment_dict) for f in self._factors))

    def randomize(self):
        for f in self._factors:
            f.randomize()

    def randomized_copy(self):
        return ProductFactor(list(f.randomized_copy() for f in self._factors))

    def mul_by_non_identity(self, other):
        if isinstance(other, ProductFactor):
            additional = other._factors
        else:
            additional = [other]
        return ProductFactor(self._factors + additional)

    def sum_out_variable(self, variable):
        factors_with_variable, factors_without_variable = split(
            self._factors, lambda f: variable in f
        )
        result_of_summing_out_variable_from_product_of_factors_with_variable = (
            Group.product(factors_with_variable) ^ variable
        )
        if factors_without_variable:
            return ProductFactor(
                factors_without_variable
                + [result_of_summing_out_variable_from_product_of_factors_with_variable]
            )
        else:
            return result_of_summing_out_variable_from_product_of_factors_with_variable

    @staticmethod
    def multiply(factors: List[Factor]) -> Factor:
        if len(factors) == 1:
            return factors[0]
        else:
            return ProductFactor(factors)

    @staticmethod
    def factors(product_of_factors):
        if isinstance(product_of_factors, ProductFactor):
            return product_of_factors._factors
        else:  # singleton
            return [product_of_factors]

    def atomic_factor(self):
        return Group.product(self._factors)

    def argmax(self):
        return self.atomic_factor().argmax()

    def normalize(self):
        return self.atomic_factor().normalize()

    def __str__(self):
        return join(self._factors, " * ")
