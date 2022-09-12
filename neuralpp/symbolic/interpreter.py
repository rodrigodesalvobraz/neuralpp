from neuralpp.symbolic.expression import Expression, Context
from abc import ABC, abstractmethod


class Interpreter(ABC):
    @abstractmethod
    def eval(self, expression: Expression, context: Context):
        """
        Evaluates the `expression` under the constraint of `context` being true.
        Either returns a value s.t., context -> expression == value
        or raise AttributeError when it's too difficult for the Interpreter; or
        raise Error encountered when applying function.
        E.g.,
        for eval(add(1,a), a==1), "return 2" and "raise AttributeError" are all legal outcomes.
        for eval(divide(1,0), True), we should expect "raise ZeroDivisionError" or "raise AttributeError".
        """
        pass
