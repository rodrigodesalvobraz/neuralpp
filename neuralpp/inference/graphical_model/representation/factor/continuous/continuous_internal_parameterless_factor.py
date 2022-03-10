import torch

from neuralpp.inference.graphical_model.representation.factor.continuous.continuous_factor import ContinuousFactor
from neuralpp.util import util
from neuralpp.util.util import join


class ContinuousInternalParameterlessFactor(ContinuousFactor):
    """
    A specialization of ContinuousFactor that does not use any internal parameters.
    """

    def __init__(self, variables, conditioning_dict=None):
        super().__init__(variables, conditioning_dict)

        # randomization does not apply since there are no internal parameters.
        self.randomize = None
        self.randomized_copy = None

    def __eq__(self, other):
        """
        Compares factors by checking if they are both NormalFactors and have the same variables and conditioning
        """
        return isinstance(other, type(self)) \
               and self.variables == other.variables \
               and self.conditioning_dict == other.conditioning_dict

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{type(self).__name__}(" + join(self.variables) + ")" + \
               (" given " + str(self.conditioning_dict) if self.conditioning_dict else "")
