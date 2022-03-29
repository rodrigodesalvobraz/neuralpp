import torch

from neuralpp.inference.graphical_model.representation.factor.continuous.pytorch_distribution_factor import \
    PyTorchDistributionFactor


class NormalFactor(PyTorchDistributionFactor):

    def __init__(self, all_variables_including_conditioned_ones, conditioning_dict=None):
        super().__init__(torch.distributions.Normal, all_variables_including_conditioned_ones, conditioning_dict)
