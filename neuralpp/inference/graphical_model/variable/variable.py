from typing import Any

import torch


class Variable:
    def __eq__(self, other) -> bool:
        self._not_implemented("__eq__")

    def __hash__(self):
        self._not_implemented("__hash__")

    def __repr__(self) -> str:
        self._not_implemented("__repr__")

    def featurize(self, value) -> torch.Tensor:
        self._not_implemented("featurize")

    def is_multivalue(self, value: Any) -> bool:
        self._not_implemented("is_multivalue")

    def value_len(self, value: Any) -> int:
        """
        Returns the length of the value if it is a multivalue, or 1 otherwise.
        """
        return self.multivalue_len(value) if self.is_multivalue(value) else 1

    def multivalue_len(self, value: Any) -> int:
        self._not_implemented("multivalue_len")

    def _not_implemented(self, name):
        error = NotImplementedError(f"{name} not implemented for {type(self)}")
        raise error
