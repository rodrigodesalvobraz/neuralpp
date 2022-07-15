from typing import Optional, Deque
from collections import deque

from .expression import Expression, Constant, FunctionApplication, QuantifierExpression
from .basic_expression import BasicQuantifierExpression, BasicFunctionApplication, BasicExpression
from .z3_expression import Z3SolverExpression
from .constants import basic_true, basic_false, if_then_else
from .context_simplifier import ContextSimplifier
from .parameters import sympy_evaluate
import neuralpp.symbolic.functions as functions


def is_non_constant_formula(expression: Expression) -> bool:
    return expression.type == bool and not isinstance(expression, Constant)  # should ignore `True` and `False`


def _bfs_first_non_constant_formula_of_expression(expression: Expression, queue: Deque) -> Optional[Expression]:
    queue.append(expression)
    while queue:
        current_expression = queue.popleft()
        if is_non_constant_formula(current_expression):
            return current_expression
        match current_expression:
            case FunctionApplication(function=Constant(value=functions.conditional), arguments=arguments):
                queue.extend(arguments[1:])  # do not include the if clause, it's meaningless
            case FunctionApplication(arguments=arguments):
                queue.extend(arguments)
            case _:
                pass  # non_constant_formulaic expression has no subexpressions
    return None


def first_non_constant_formula_of_expression(expression: Expression) -> Optional[Expression]:
    """ returns the first non_constant_formula occurring in a given expression in a breadth-first search. """
    return _bfs_first_non_constant_formula_of_expression(expression, deque())


class Normalizer:
    simplifier = ContextSimplifier()

    @staticmethod
    def _basic_normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
        """
        The word `basic` in the function name is not to be confused with that in `BasicExpression`.
        Here it refers to normalization of non-quantifier expressions.
        The function assumes context is satisfiable, otherwise will raise (from Normalizer.simplifier.simplify()).
        """
        simplified_expression = Normalizer.simplifier.simplify(expression, context)
        # We look for non-constant formula, because we should not split on constants (i.e., True/False)
        # as it is meaningless and would cause the algorithm to not terminate.
        # E.g., split on `True` would result in `if True then True else False`.
        first_non_constant_formula = first_non_constant_formula_of_expression(simplified_expression)
        if first_non_constant_formula is None:
            return simplified_expression

        context1 = context & first_non_constant_formula
        # The following statement can be proved by contradiction:
        # if not isinstance(context1, Z3SolverExpression), then context & first_non_constant_formula must be False,
        # then context -> ~first_non_constant_formula, then in Normalizer.simplifier.simplify(expression, context),
        # first_non_constant_formula as a subexpression of expression should have been replaced by False,
        # since its negation is implied by context, thus should not be a subexpression of simplified_expression. Q.E.D.
        assert isinstance(context1, Z3SolverExpression)
        expression1 = simplified_expression.replace(first_non_constant_formula, basic_true)
        then_clause = Normalizer._normalize(expression1, context1)

        context2 = context & ~first_non_constant_formula
        assert isinstance(context2, Z3SolverExpression)
        expression2 = simplified_expression.replace(first_non_constant_formula, basic_false)
        else_clause = Normalizer._normalize(expression2, context2)

        return if_then_else(first_non_constant_formula, then_clause, else_clause)

    @staticmethod
    def _normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
        """ The function assumes context is satisfiable, otherwise will raise. """
        if isinstance(expression, QuantifierExpression):
            return Normalizer._quantifier_expression_normalize(expression, context)
        else:
            return Normalizer._basic_normalize(expression, context)

    @staticmethod
    def normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
        with sympy_evaluate(True):
            return Normalizer._normalize(expression, context)

    @staticmethod
    def _quantifier_expression_normalize(expression: QuantifierExpression, context: Z3SolverExpression) -> Expression:
        """ The function assumes context is satisfiable, otherwise will raise. """
        if context.structurally_contains(expression.index):
            raise ValueError(f"{context} should not contain {expression.index}, which is not a free variable!")

        normalized_body = Normalizer._normalize(expression.body, context)
        match normalized_body:
            case FunctionApplication(function=function, arguments=arguments,
                                     ) if function.value == functions.conditional:
                condition, then_clause, else_clause = arguments  # if len(arguments) != 3, raise
                if condition.structurally_contains(expression.index):
                    new_then_clause = BasicExpression.new_quantifier_expression(
                        expression.operation, expression.index, expression.constrain & condition, then_clause)
                    new_else_clause = BasicExpression.new_quantifier_expression(
                        expression.operation, expression.index, expression.constrain & ~condition, else_clause)
                    return BasicFunctionApplication(expression.operation, [new_then_clause, new_else_clause])
                else:
                    return BasicFunctionApplication(function, [condition,
                                                               expression.set_body(then_clause),
                                                               expression.set_body(else_clause)])
            case _:
                return expression.set_body(normalized_body)
