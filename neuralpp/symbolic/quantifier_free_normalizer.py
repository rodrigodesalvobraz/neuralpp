from typing import Optional, Deque
from collections import deque

from .expression import Expression, Constant, FunctionApplication
from .z3_expression import Z3SolverExpression
from .constants import basic_true, basic_false, if_then_else
from .context_simplifier import ContextSimplifier
from .parameters import sympy_evaluate
from .normalizer import Normalizer


def is_non_constant_formula(expression: Expression) -> bool:
    return expression.type == bool and not isinstance(
        expression, Constant
    )  # should ignore `True` and `False`


def _bfs_first_non_constant_formula_of_expression(
    expression: Expression, queue: Deque
) -> Optional[Expression]:
    queue.append(expression)
    while queue:
        current_expression = queue.popleft()
        if is_non_constant_formula(current_expression):
            return current_expression
        match current_expression:
            case FunctionApplication(arguments=arguments):
                # This includes if clause in a conditional expression, we need to split on that
                # to simplify its subexpressions.
                # E.g. if x > 3 then (if x > 2 then .. else ..) else ..
                # if we do not split on x > 3, then the simplification x > 2 -> True cannot be made
                for argument in arguments:
                    queue.append(argument)
            case _:
                pass  # non_constant_formulaic expression has no subexpressions
    return None


def first_non_constant_formula_of_expression(
    expression: Expression,
) -> Optional[Expression]:
    """returns the first non_constant_formula occurring in a given expression in a breadth-first search."""
    assert isinstance(expression, Expression)
    return _bfs_first_non_constant_formula_of_expression(expression, deque())


class QuantifierFreeNormalizer(Normalizer):
    simplifier = ContextSimplifier()

    @staticmethod
    def _normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
        simplified_expression = QuantifierFreeNormalizer.simplifier.simplify(
            expression, context
        )
        # We look for non-constant formula, because we should not split on constants (i.e., True/False)
        # as it is meaningless and would cause the algorithm to not terminate.
        # E.g., split on `True` would result in `if True then True else False`.
        first_non_constant_formula = first_non_constant_formula_of_expression(
            simplified_expression
        )
        if first_non_constant_formula is None:
            return simplified_expression

        context1 = context & first_non_constant_formula
        # The following statement can be proved by contradiction:
        # if not isinstance(context1, Z3SolverExpression), then context & first_non_constant_formula must be False,
        # then context -> ~first_non_constant_formula, then in Normalizer.simplifier.simplify(expression, context),
        # first_non_constant_formula as a subexpression of expression should have been replaced by False,
        # since its negation is implied by context, thus should not be a subexpression of simplified_expression. Q.E.D.
        assert isinstance(context1, Z3SolverExpression)
        expression1 = simplified_expression.replace(
            first_non_constant_formula, basic_true
        )
        then_clause = QuantifierFreeNormalizer._normalize(expression1, context1)

        context2 = context & ~first_non_constant_formula
        assert isinstance(context2, Z3SolverExpression)
        expression2 = simplified_expression.replace(
            first_non_constant_formula, basic_false
        )
        else_clause = QuantifierFreeNormalizer._normalize(expression2, context2)

        return if_then_else(first_non_constant_formula, then_clause, else_clause)

    @staticmethod
    def normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
        with sympy_evaluate(True):
            return QuantifierFreeNormalizer._normalize(expression, context)
