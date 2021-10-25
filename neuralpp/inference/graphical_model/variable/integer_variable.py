from typing import Any

import torch
from neuralpp.inference.graphical_model.variable.discrete_variable import (
    DiscreteVariable,
)
from neuralpp.util.util import is_iterable


class IntegerVariable(DiscreteVariable):
    def __init__(self, name, cardinality):
        super().__init__(name, cardinality)

    def assignments(self):
        return range(self.cardinality)

    def featurize(self, integer_value) -> torch.Tensor:
        if self.is_multivalue(integer_value):
            return torch.tensor(integer_value, dtype=torch.float).unsqueeze(1)
        else:
            return torch.tensor([integer_value], dtype=torch.float)

    def is_multivalue(self, value: Any) -> bool:
        return is_iterable(value)

    def multivalue_len(self, value: Any) -> int:
        assert self.is_multivalue(value)
        return len(value)

    def __eq__(self, other):
        assert isinstance(
            other, IntegerVariable
        ), "IntegerVariable being compared to non-IntegerVariable"
        result = self.name == other.name
        if result and self.cardinality != other.cardinality:
            print(
                f"Warning: variables {self} and {other} have the same name but different"
                f"cardinalities {self.cardinality} and {other.cardinality}"
            )
        return result

    def __hash__(self):
        return hash(self.name) + 31 * self.cardinality

    def __repr__(self):
        return str(self.name)
