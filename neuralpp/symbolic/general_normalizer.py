from .normalizer import Normalizer
from .context_simplifier import ContextSimplifier
from .expression import Expression, Constant, Variable, FunctionApplication, QuantifierExpression, Context, \
    AbelianOperation
from .z3_expression import Z3SolverExpression
from .constants import basic_true, basic_false, if_then_else
from .parameters import sympy_evaluate
from .basic_expression import BasicQuantifierExpression, BasicExpression
import neuralpp.symbolic.functions as functions
from typing import List


def _normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
    """
    The function assumes context is satisfiable, otherwise will raise.
    The input expression can be arbitrarily structured. The result expression is guaranteed to be
    `quantifiers-at-leaves`: quantifiers are at leaves and do not contain conditional function in their bodies.
    """
    match expression:
        case Constant():
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
        case Expression(type=type_) if type_ == bool:
            if context.is_known_to_imply(expression):
                return basic_true
            if context.is_known_to_imply(~expression):
                return basic_false
            return if_then_else(expression, True, False)
        case Variable():
            return expression
        case FunctionApplication(function=function, arguments=arguments):
            return _normalize_function_application(function, arguments, context)
        case QuantifierExpression(operation=operation, index=index, constraint=constraint, body=body):
            if context.contains(index):
                raise ValueError(f"context {context} should not contain index {index}")
            normalized_body = _normalize(body, context & constraint)
            match normalized_body:
                case FunctionApplication(function=Constant(value=functions.conditional),
                                         arguments=[condition, then, else_]):
                    if condition.contains(index):
                        return _normalize(
                            operation(BasicQuantifierExpression(operation, index, constraint & condition, then),
                                      BasicQuantifierExpression(operation, index, constraint & ~condition, else_)),
                            context)
                    else:
                        return if_then_else(condition,
                                            _normalize(BasicQuantifierExpression(operation, index, constraint, then),
                                                       context & condition),
                                            _normalize(BasicQuantifierExpression(operation, index, constraint, else_),
                                                       context & ~condition))
                case _:
                    return _eliminate(operation, index, constraint, normalized_body, context)


_simplifier = ContextSimplifier()


def _eliminate(operation: AbelianOperation, index: Variable, constraint: Context, body: Expression,
               context: Z3SolverExpression) -> Expression:
    """
    Eliminates all quantifiers by doing the "summation" (or to use a Computer Science term, "reduction").
    Currently, a very naive placeholder.
    In particular, we allow eliminate() to accept body and return expression that contains quantifier,
    thus it is a "partial/best-effort elimination".
    In particular, we expect body and result to be normalized quantifiers-at-leaves.

    Future work: more complicated elimination algorithm, which actually tries to `eliminate` quantifiers.
    In general 2 directions for improvement:
    1. supports more operations (add, multiply, and, or, ...)
    2. supports multiple intervals & complicated constraints (e.g, 1 <= x <= 100, x != y)
    """
    if context.is_known_to_imply(~constraint):
        return operation.identity
    return BasicQuantifierExpression(operation, index, constraint, _simplifier.simplify(body, context & constraint))


def _normalize_function_application(function: Expression,
                                    arguments: List[Expression],
                                    context: Z3SolverExpression, i: int = 0) -> Expression:
    if i >= len(arguments):
        return function(*arguments)
    n_i = _normalize(arguments[i], context)
    arguments[i] = n_i
    return _move_down_and_normalize(function, arguments, context, i)


def _move_down_and_normalize(function: Expression,
                             arguments: List[Expression],
                             context: Z3SolverExpression, i: int) -> Expression:
    """
    assume i < len(arguments);
    assume all {arguments[j] | j < i} does not contain if-then-else (and thus normalized);
    assume arguments[i] has been normalized.
    move down the f to the leaves. Then replace every leaf f with its normalized version
    """
    match arguments[i]:
        case FunctionApplication(function=Constant(value=functions.conditional),
                                 arguments=[condition, then, else_]):
            then_arguments = arguments[:]
            then_arguments[i] = then
            else_arguments = arguments[:]
            else_arguments[i] = else_
            return if_then_else(condition,
                                _move_down_and_normalize(function, then_arguments, context & condition, i),
                                _move_down_and_normalize(function, else_arguments, context & ~condition, i))
        case _:
            return _normalize_function_application(function, arguments, context, i + 1)


class GeneralNormalizer(Normalizer):
    @staticmethod
    def normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
        with sympy_evaluate(True):
            return _normalize(expression, context)
