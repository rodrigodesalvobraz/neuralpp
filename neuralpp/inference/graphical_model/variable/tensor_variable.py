import functools
from typing import Optional, Tuple, Union, Any

import torch

from neuralpp.inference.graphical_model.variable.variable import Variable

FlexibleShape = Union[torch.Size, Tuple[int, ...]]


class TensorVariable(Variable):
    def __init__(self, name, non_batch_dim: Optional[int] = None):
        """
        Constructs a tensor variable with given name and optional non-batch shape.

        Non-batch shape is used for deciding whether a value should be considered a batch or not.
        In some applications this is not needed, so this information is optional.
        An exception will be raised if this is not provided but a functionality
        dependent on it is needed.
        """
        super().__init__()
        self.name = name
        self.non_batch_dim = non_batch_dim

    def featurize(self, value) -> torch.Tensor:
        self._check_value_is_tensor(value)
        return value

    def is_multivalue(self, value: Any) -> bool:
        self._check_value_is_tensor(value)
        self._check_value_dimension(value)
        return value.dim() == self.non_batch_dim + 1

    def multivalue_len(self, value: Any) -> int:
        assert self.is_multivalue(value)
        return value.shape[0]

    @staticmethod
    def _check_value_is_tensor(value):
        if not isinstance(value, torch.Tensor):
            raise Exception(
                f"Values of {TensorVariable.__name__} must be tensors."
            )

    def _check_value_dimension(self, tensor):
        if tensor.dim() not in {self.non_batch_dim, self.non_batch_dim + 1}:
            raise Exception(
                f"Tensor value for {self} must have dimension "
                f"{self.non_batch_dim} or {self.non_batch_dim + 1} (if batch)"
            )

    def __eq__(self, other):
        if not isinstance(other, TensorVariable):
            return False
        result = self.name == other.name
        return result

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self.name)
