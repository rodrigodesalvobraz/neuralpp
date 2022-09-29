from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
)
from neuralpp.util import util


class AtomicFactor(Factor):
    def __init__(self, variables):
        super().__init__(variables)

    def atomic_factor(self):
        return self

    def sample(self, n=1):
        self._not_implemented("sample")

    # Convenience methods

    def sample_assignment_dict(self, n=1):
        return self.from_assignment_to_assignment_dict(self.sample(n))
