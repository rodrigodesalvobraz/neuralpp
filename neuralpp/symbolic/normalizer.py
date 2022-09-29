from abc import abstractmethod, ABC
from .expression import Expression
from .z3_expression import Z3SolverExpression


class Normalizer(ABC):
    @staticmethod
    @abstractmethod
    def normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
        pass
