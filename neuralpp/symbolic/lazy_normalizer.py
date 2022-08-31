from .normalizer import Normalizer
from .expression import Expression, Constant, Variable, FunctionApplication, QuantifierExpression
from .z3_expression import Z3SolverExpression
from .constants import if_then_else
import neuralpp.symbolic.functions as functions
from .general_normalizer import GeneralNormalizer
from .profiler import Profiler


class LazyNormalizer(Normalizer):
    def __init__(self, profiler: Profiler = None):
        if profiler is None:
            self.profiler = Profiler()
        else:
            self.profiler = profiler
        self._general_normalizer = GeneralNormalizer(self.profiler)

    def normalize(self, expression: Expression, context: Z3SolverExpression) -> Expression:
        with self.profiler.profile_section("normalization"):
            return self._normalize(expression, context)

    def _normalize(self, expression: Expression, context: Z3SolverExpression) -> Expression:
        match expression:
            case Constant() | Variable():
                return expression
            case FunctionApplication(function=Constant(value=functions.conditional),
                                     arguments=[condition, then, else_]):
                if context.is_known_to_imply(condition):
                    return self._normalize(then, context)
                elif context.is_known_to_imply(~condition):
                    return self._normalize(else_, context)
                else:
                    return if_then_else(condition,
                                        self._normalize(then, context & condition),
                                        self._normalize(else_, context & ~condition))
            case FunctionApplication(function=function, arguments=arguments):
                arguments = [self._normalize(argument, context) for argument in arguments]
                return expression.new_function_application(function, arguments)
            case QuantifierExpression():
                return self._general_normalizer.normalize(expression, context)
            case _:
                raise ValueError(f"invalid expression {expression}")
