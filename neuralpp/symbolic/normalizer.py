from typing import Optional, Deque
from collections import deque

from .expression import Expression, Constant, FunctionApplication
from .z3_expression import Z3SolverExpression
from .constants import basic_true, basic_false, if_then_else
from .context_simplifier import ContextSimplifier
from .parameters import sympy_evaluate
import neuralpp.symbolic.functions as functions


def is_boolean_atom(expression: Expression) -> bool:
    return expression.type == bool and not isinstance(expression, Constant)  # should ignore `True` and `False`


def _bfs_first_atom_of_expression(expression: Expression, queue: Deque) -> Optional[Expression]:
    queue.append(expression)
    while queue:
        current_expression = queue.popleft()
        if is_boolean_atom(current_expression):
            return current_expression
        match current_expression:
            case FunctionApplication(function=Constant(value=functions.conditional), arguments=arguments):
                queue.extend(arguments[1:])  # do not include the if clause, it's meaningless
            case FunctionApplication(arguments=arguments):
                queue.extend(arguments)
            case _:
                pass  # atomic expression has no subexpressions
    return None


def first_atom_of_expression(expression: Expression) -> Optional[Expression]:
    """ returns the first atom occurring in a given expression in a breadth-first search. """
    return _bfs_first_atom_of_expression(expression, deque())


class Normalizer:
    simplifier = ContextSimplifier()

    @staticmethod
    def _normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
        simplified_expression = Normalizer.simplifier.simplify(expression, context)
        first_atom = first_atom_of_expression(simplified_expression)
        if first_atom is None:
            return simplified_expression

        context1 = context & first_atom
        # The following statement can be proved by contradiction:
        # if not isinstance(context1, Z3SolverExpression), then context & first_atom must be False,
        # then context -> ~first_atom, then in Normalizer.simplifier.simplify(expression, context),
        # first_atom as a subexpression of expression should have been replaced by False since its negation
        # is implied by context, thus should not be a subexpression of simplified_expression. Q.E.D.
        assert isinstance(context1, Z3SolverExpression)
        expression1 = simplified_expression.replace(first_atom, basic_true)
        then_clause = Normalizer._normalize(expression1, context1)

        context2 = context & ~first_atom
        assert isinstance(context2, Z3SolverExpression)
        expression2 = simplified_expression.replace(first_atom, basic_false)
        else_clause = Normalizer._normalize(expression2, context2)

        return if_then_else(first_atom, then_clause, else_clause)

    @staticmethod
    def normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
        with sympy_evaluate(True):
            return Normalizer._normalize(expression, context)
