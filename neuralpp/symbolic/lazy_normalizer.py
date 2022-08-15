from .normalizer import Normalizer
from .expression import Expression, Constant, Variable, FunctionApplication, QuantifierExpression
from .z3_expression import Z3SolverExpression
from .constants import if_then_else
from .parameters import sympy_evaluate
import neuralpp.symbolic.functions as functions
from .general_normalizer import GeneralNormalizer
from .evaluator import Evaluator


class LazyNormalizer(Normalizer):
    def __init__(self, evaluator: Evaluator = None):
        if evaluator is None:
            self.evaluator = Evaluator()
        else:
            self.evaluator = evaluator
        self._general_normalizer = GeneralNormalizer(self.evaluator)

    def normalize(self, expression: Expression, context: Z3SolverExpression) -> Expression:
        match expression:
            case Constant() | Variable():
                return expression
            case FunctionApplication(function=Constant(value=functions.conditional),
                                     arguments=[condition, then, else_]):
                if context.is_known_to_imply(condition):
                    return self.normalize(then, context)
                elif context.is_known_to_imply(~condition):
                    return self.normalize(else_, context)
                else:
                    return if_then_else(condition,
                                        self.normalize(then, context & condition),
                                        self.normalize(else_, context & ~condition))
            case FunctionApplication(function=function, arguments=arguments):
                arguments = [self.normalize(argument, context) for argument in arguments]
                return expression.new_function_application(function, arguments)
            case QuantifierExpression():
                return self._general_normalizer.normalize(expression, context)
            case _:
                raise ValueError(f"invalid expression {expression}")

