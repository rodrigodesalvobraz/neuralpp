import torch

from neuralpp.inference.graphical_model.representation.factor.continuous.continuous_factor import ContinuousFactor
from neuralpp.util import util
from neuralpp.util.util import join


class NormalFactor(ContinuousFactor):

    def __init__(self, x, mu, sigma, conditioning_dict=None):
        super().__init__([x, mu, sigma])
        self.x = x
        self.mu = mu
        self.sigma = sigma
        self.conditioning_dict = conditioning_dict

        # randomization does not apply as Normal has no internal parameters.
        self.randomize = None
        self.randomized_copy = None

    def condition_on_non_empty_dict(self, assignment_dict):
        return NormalFactor(self.x, self.mu, self.sigma, self.total_conditioning_dict(assignment_dict))

    def call_after_validation(self, assignment_dict, assignment_values):
        assignment_and_conditioning_dict = self.total_conditioning_dict(assignment_dict)
        mu = assignment_and_conditioning_dict[self.mu]
        sigma = assignment_and_conditioning_dict[self.sigma]
        normal = torch.distributions.Normal(loc=mu, scale=sigma)
        x = assignment_and_conditioning_dict[self.x]
        return normal.log_prob(x).exp()

    def total_conditioning_dict(self, assignment_dict):
        return util.union_of_dicts(
            assignment_dict, self.conditioning_dict
        )

    def __eq__(self, other):
        """
        Compares factors by checking if they are both NormalFactors and have the same variables and conditioning
        """
        return isinstance(other, NormalFactor) \
               and self.variables == other.variables \
               and self.conditioning_dict == other.conditioning_dict

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "NormalFactor(" + join(self.variables) + ")" + \
               (" given " + self.conditioning_dict if self.conditioning_dict else "")
