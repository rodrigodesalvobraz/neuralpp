import torch

from neuralpp.inference.graphical_model.representation.factor.continuous.pytorch_distribution_factor import \
    PyTorchDistributionFactor


class NormalFactor(PyTorchDistributionFactor):

    def __init__(self, variables, conditioning_dict=None):
        super().__init__(torch.distributions.Normal, variables, conditioning_dict)
