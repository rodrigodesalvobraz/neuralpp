from .normalizer import Normalizer
from .context_simplifier import ContextSimplifier
from .expression import Expression, Constant, Variable, FunctionApplication, QuantifierExpression, Context, \
    AbelianOperation
from .z3_expression import Z3SolverExpression
from .constants import basic_true, basic_false, if_then_else
from .parameters import sympy_evaluate
from .basic_expression import BasicQuantifierExpression, BasicExpression
from .eliminator import Eliminator
import neuralpp.symbolic.functions as functions
from typing import List


_simplifier = ContextSimplifier()
_eliminator = Eliminator()


def _normalize_conditional(condition: Expression, then: Expression, else_: Expression, context: Z3SolverExpression):
    """
    Arguments condition-then-else_ forms an if-then-else statement.
    If either branch can be pruned (its negation implied by context), it is pruned.
    It is not just an optimization but a requirement that every prunable be pruned:
    otherwise we'd call normalize() with an unsatisfiable context, which is not well-defined.
    """
    if context.is_known_to_imply(condition):
        return _normalize(then, context)
    elif context.is_known_to_imply(~condition):
        return _normalize(else_, context)
    else:
        return if_then_else(condition,
                            _normalize(then, context & condition),
                            _normalize(else_, context & ~condition))


def _normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
    """
    The function assumes context is satisfiable, otherwise the behavior is undefined.
    The input expression can be arbitrarily structured. The result expression is guaranteed to be
    `quantifiers-at-leaves`: quantifiers are at leaves and do not contain conditional function in their bodies.
    """
    assert not context.unsatisfiable
    match expression:
        case Constant():
            return expression
        case FunctionApplication(function=Constant(value=functions.conditional),
                                 arguments=[condition, then, else_]):
            return _normalize_conditional(condition, then, else_, context)
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
            if context.is_known_to_imply(~constraint):
                return operation.identity
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
                        return _normalize_conditional(condition,
                                                      BasicQuantifierExpression(operation, index, constraint, then),
                                                      BasicQuantifierExpression(operation, index, constraint, else_),
                                                      context)
                case _:
                    return _eliminate(operation, index, constraint, normalized_body, context)


def _eliminate(operation: AbelianOperation, index: Variable, constraint: Context, body: Expression,
               context: Z3SolverExpression) -> Expression:
    """
    Eliminates all quantifiers by doing the "summation" (or to use a Computer Science term, "reduction").
    In particular, we expect body and result to be normalized quantifiers-at-leaves.

    Future work: more complicated elimination algorithm, which actually tries to `eliminate` quantifiers.
    In general 2 directions for improvement:
    1. supports more operations (add, multiply, and, or, ...)
    2. supports multiple intervals & complicated constraints (e.g, 1 <= x <= 100, x != y)
    """
    if context.is_known_to_imply(~constraint):
        return operation.identity
    return _eliminator.eliminate(operation, index, constraint, body, context)


def _normalize_function_application(function: Expression,
                                    arguments: List[Expression],
                                    context: Z3SolverExpression, i: int = 0) -> Expression:
    if i >= len(arguments):
        return _simplifier.simplify(function(*arguments), context)
    arguments[i] = _normalize(arguments[i], context)
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
