from neuralpp.inference.graphical_model.representation.factor.factor import Factor
from neuralpp.util import util


class AtomicFactor(Factor):
    def __init__(self, variables):
        super().__init__(variables)

    def atomic_factor(self):
        return self

    def sample(self):
        self._not_implemented("sample")

    def single_sample(self):
        """TODO: deprecated; remove"""
        self._not_implemented("single_sample")

    # Convenience methods

    def single_sample_assignment_dict(self):
        """TODO: deprecated; remove"""
        sampled_assignment = self.single_sample()
        return self.from_assignment_to_assignment_dict(sampled_assignment)
