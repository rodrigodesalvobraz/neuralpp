import torch

from neuralpp.inference.graphical_model.variable.variable import Variable


class TensorVariable(Variable):

    def __init__(self, name):
        super().__init__()
        self.name = name

    def featurize(self, pytorch_tensor_value):
        assert isinstance(pytorch_tensor_value, torch.Tensor), f"Values of {TensorVariable.__name__} must be tensors. "\
                                                               f"Moreover, they must be batch tensors even if they "\
                                                               f"contain is a single value."
        return pytorch_tensor_value

    def __eq__(self, other):
        assert isinstance(other, TensorVariable), "TensorVariable being compared to non-TensorVariable"
        result = self.name == other.name
        return result

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self.name)
