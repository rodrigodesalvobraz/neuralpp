from .normalizer import Normalizer
from .expression import Expression, Constant, Variable, FunctionApplication, QuantifierExpression
from .z3_expression import Z3SolverExpression
from .constants import if_then_else
from .parameters import sympy_evaluate
import neuralpp.symbolic.functions as functions
from .general_normalizer import GeneralNormalizer


def _normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
    match expression:
        case Constant() | Variable():
            return expression
        case FunctionApplication(function=Constant(value=functions.conditional),
                                 arguments=[condition, then, else_]):
            if context.is_known_to_imply(condition):
                return _normalize(then, context)
            elif context.is_known_to_imply(~condition):
                return _normalize(else_, context)
            else:
                return if_then_else(condition,
                                    _normalize(then, context & condition),
                                    _normalize(else_, context & ~condition))
        case FunctionApplication(function=function, arguments=arguments):
            arguments = [_normalize(argument, context) for argument in arguments]
            return expression.new_function_application(function, arguments)
        case QuantifierExpression():
            return GeneralNormalizer.normalize(expression, context)
        case _:
            raise ValueError(f"invalid expression {expression}")


class LazyNormalizer(Normalizer):
    @staticmethod
    def normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
        with sympy_evaluate(True):
            return _normalize(expression, context)
