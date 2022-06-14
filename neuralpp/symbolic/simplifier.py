from abc import ABC, abstractmethod
from neuralpp.symbolic.expression import Expression


class Simplifier(ABC):
    @abstractmethod
    def simplify(self, expression: Expression, context: Expression) -> Expression:
        pass
