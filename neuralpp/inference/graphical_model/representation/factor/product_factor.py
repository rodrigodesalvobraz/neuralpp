import math

from neuralpp.inference.graphical_model.representation.factor.factor import Factor
from neuralpp.util.group import Group
from neuralpp.util.util import join, split


class ProductFactor(Factor):
    def __init__(self, factors):
        super().__init__(set.union(*(set(f.variables) for f in factors)))
        self._factors = factors

    def call_after_validation(self, assignment_dict, assignment_values):
        return math.prod(f(assignment_dict) for f in self._factors)

    def condition_on_non_empty_dict(self, assignment_dict):
        return ProductFactor(f.condition(assignment_dict) for f in self._factors)

    def randomize(self):
        for f in self._factors:
            f.randomize()

    def randomized_copy(self):
        return ProductFactor(f.randomized_copy() for f in self._factors)

    def __mul__(self, other):
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
            # at this point, result_of_summing_out_variable_from_product_of_factors_with_variable
            # may be a ProductFactor or an atomic factor.
            # Here we ensure it is a ProductFactor
            return ProductFactor.make(
                result_of_summing_out_variable_from_product_of_factors_with_variable
            )

    @staticmethod
    def make(factor):
        return ProductFactor(ProductFactor.factors(factor))

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
