from abc import ABC, abstractmethod
from neuralpp.symbolic.expression import Expression, Context


class Simplifier(ABC):
    @abstractmethod
    def simplify(
        self, expression: Expression, context: Context
    ) -> Expression:
        pass
