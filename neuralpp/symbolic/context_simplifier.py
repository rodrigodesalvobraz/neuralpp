from typing import Optional, Callable, Set

from neuralpp.symbolic.constants import basic_true, basic_false
from neuralpp.symbolic.expression import Expression
from neuralpp.symbolic.expression import QuantifierExpression
from neuralpp.symbolic.parameters import sympy_evaluate
from neuralpp.symbolic.simplifier import Simplifier
from neuralpp.symbolic.sympy_interpreter import SymPyInterpreter
from neuralpp.symbolic.z3_expression import Z3SolverExpression
from neuralpp.util.symbolic_error_util import ConversionError, UnknownError


def _simplify_expression(
    boolean_expression: Expression, context: Z3SolverExpression
) -> Optional[Expression]:
    if boolean_expression.type != bool:
        raise TypeError("Can only simplify booleans")
    if context.is_known_to_imply(boolean_expression):
        return basic_true
    if context.is_known_to_imply(~boolean_expression):
        return basic_false
    return None


def _collect_subset_expressions_helper(
    expression: Expression, test: Callable[[Expression], bool], result: Set[Expression]
):
"""
Checks to see if `expression` fullfils the filter `test`
Also, recursively calls itself on the possible subexpressions in `expression`
"""
    if test(expression):
        result.add(expression)
    for sub_expression in expression.subexpressions:
        _collect_subset_expressions_helper(sub_expression, test, result)


def _collect_subset_expressions(
    expression: Expression,
    test: Callable[[Expression], bool] = lambda _: True,
) -> Set[Expression]:
    """
    Collect all subexpressions including the ones that are not immediate subexpressions and self, with a filter
    `test`, and adds them into a Set.
    """
    result = set()
    _collect_subset_expressions_helper(expression, test, result)
    return result


class ContextSimplifier(Simplifier):
    sympy_interpreter = SymPyInterpreter()

    @staticmethod
    def _simplify_pass(
        expression: Expression, context: Z3SolverExpression
    ) -> Expression:
        try:
            result = ContextSimplifier.sympy_interpreter.simplify(expression, context)
        except ConversionError:
            result = expression

        # replace boolean expressions
        all_boolean_subexpressions = _collect_subset_expressions(
            result,
            lambda expr: expr.type == bool
            and not isinstance(expr, QuantifierExpression),
        )
        for boolean_subexpression in all_boolean_subexpressions:
            simplified_subexpression = _simplify_expression(
                boolean_subexpression, context
            )
            if simplified_subexpression is not None:
                result = result.replace(boolean_subexpression, simplified_subexpression)
                if result is None:
                    raise RuntimeError("result is None")

        # replace variables
        for variable, replacement in context.variable_replacement_dict.items():
            result = result.replace(variable, replacement)
        return result

    def simplify(
        self, expression: Expression, context: Z3SolverExpression
    ) -> Expression:
        with sympy_evaluate(
            True
        ):  # To work around a bug in SymPy (see context_simplifier_test.py/test_sympy_bug).
            if not isinstance(context, Z3SolverExpression):
                raise ValueError(
                    "ContextSimplifier expects a Z3SolverExpression context."
                )
            if not context.satisfiability_is_known:
                raise UnknownError()
            if context.unsatisfiable:
                raise ValueError(f"Context {context} is unsatisfiable.")

            new_expression = ContextSimplifier._simplify_pass(expression, context)
            while not new_expression.syntactic_eq(expression):
                expression = new_expression
                new_expression = ContextSimplifier._simplify_pass(expression, context)
            return new_expression
