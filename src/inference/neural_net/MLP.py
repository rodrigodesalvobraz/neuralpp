import torch
from torch import nn
from torch.nn import Module

from util import util
from util.util import pairwise, vararg_or_array, join


class MLP(Module):
    """
    A multi-layer perceptron composed of a configurable number of linear layers followed by sigmoid transformations.
    The last layer output elements will numbers between 0 and 1.
    They do not necessarily sum up to 1; each must be considered a separate probability.
    """

    def __init__(self, *all_layer_sizes):
        super(MLP, self).__init__()
        self.layer_sizes = vararg_or_array(all_layer_sizes)
        assert len(self.layer_sizes) > 0
        assert all(isinstance(layer_size, int) for layer_size in self.layer_sizes), f"Layer sizes must be ints but got a layer size value equal to {util.find(self.layer_sizes, lambda e: not isinstance(e, int))}"
        assert all(layer_size > 0 for layer_size in self.layer_sizes), f"MLP layers must have sizes greater than zero, but were {all_layer_sizes}"
        self.linear_transformations = nn.ModuleList([torch.nn.Linear(i, o) for i, o in pairwise(self.layer_sizes)])
        self.last_layer_index = len(self.linear_transformations) - 1

    def forward(self, x):
        assert isinstance(x, torch.Tensor)
        if not torch.is_floating_point(x):
            x = x.float()
        for i, linear_transformation in enumerate(self.linear_transformations):
            transformation = linear_transformation(x)

            # print("----------------------")
            # print(rounded_list(x))
            # print("*")
            # print(rounded_list(linear_transformation.weight))
            # print("+")
            # print(rounded_list(linear_transformation.bias))
            # print("=")
            # print(rounded_list(transformation))

            if i == self.last_layer_index:
                x = torch.nn.functional.softmax(transformation, dim=-1)
                # print("given to softmax ->")
                # print(rounded_list(x))
            else:
                x = torch.sigmoid(transformation)
                # print("given to sigmoid ->")
                # print(rounded_list(x))
        return x

    def randomize(self):
        for layer in self.linear_transformations:
            # we pick a random number in [-5.0, 5.0] because that is the range needed
            # to ensure sigmoid produces values from near 0 to near 1,
            # divided by number of inputs so it does not saturate when there are many inputs
            n_inputs = layer.weight.shape[1] + 1
            uniform_range = 5.0/n_inputs
            torch.nn.init.uniform_(layer.weight, -uniform_range, uniform_range)
            torch.nn.init.uniform_(layer.bias, -uniform_range, uniform_range)

            # print("n_inputs:", n_inputs)
            # print("Randomized layer.weight:", layer.weight)
            # print("Randomized layer.bias:", layer.bias)

    def randomized_copy(self):
        result = MLP(self.layer_sizes)
        result.randomize()
        return result

    def __repr__(self):
        return f"MLP({join(self.layer_sizes)})"
