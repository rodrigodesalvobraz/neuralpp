from neuralpp.inference.graphical_model.representation.factor.factor import Factor
from neuralpp.util import util


class AtomicFactor(Factor):
    def __init__(self, variables):
        super().__init__(variables)

    def atomic_factor(self):
        return self

    def sample(self, n=1):
        """
        Returns a sampled tensor (or batch thereof) of indices according to the factor probabilities.
        The batch dimensions will correspond to the factor.batch and number of samples if different from 1.
        For example, the resulting tensor will have three dimensions if the factor is batched and n > 1,
        but only two if either the factor is batched and n = 1, or the factor is not batched and n > 1.
        """
        self._not_implemented("sample")

    # Convenience methods

    def sample_assignment_dict(self, n=1):
        return self.from_assignment_to_assignment_dict(self.sample(n))
