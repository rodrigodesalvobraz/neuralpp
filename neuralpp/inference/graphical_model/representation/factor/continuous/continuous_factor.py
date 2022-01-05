import functools

import torch

from neuralpp.inference.graphical_model.representation.factor.atomic_factor import AtomicFactor
from neuralpp.inference.graphical_model.representation.factor.product_factor import ProductFactor
from neuralpp.util import util
from neuralpp.util.util import join


class ContinuousFactor(AtomicFactor):
    """An abstract specialization of AtomicFactor with at least one continuous variable"""

    def __init__(self, variables, conditioning_dict=None):
        super().__init__(variables)
        self.conditioning_dict = conditioning_dict if conditioning_dict else {}

    def total_conditioning_dict(self, assignment_dict):
        return util.union_of_dicts(
            assignment_dict, self.conditioning_dict
        )

    def assignments(self):
        self._invalid_method("assignments")

    @property
    @functools.lru_cache(1)
    def table_factor(self):
        self._invalid_method("table_factor")

    @staticmethod
    def _invalid_method(method_name):
        error = NotImplementedError(f"Factor.{method_name} invalid for factors with continuous variables")
        raise error

    def mul_by_non_identity(self, other):
        return ProductFactor(self, other)

    def sum_out_variable(self, variable):
        self._not_implemented("sum_out_variable")

    def argmax(self):
        self._not_implemented("argmax")

    def normalize(self):
        self._not_implemented("normalize")

    def sample(self):
        self._not_implemented("sample")

    def _not_implemented(self, method_name):
        error = NotImplementedError(f"Factor.{method_name} not yet supported for {type(self)}")
        raise error
