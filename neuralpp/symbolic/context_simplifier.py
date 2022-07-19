from typing import Optional, Callable, Set
from .expression import Expression, Context, ConversionError
from .simplifier import Simplifier
from .sympy_expression import SymPyExpression
from .sympy_interpreter import SymPyInterpreter
from .constants import basic_true, basic_false
from .z3_expression import Z3SolverExpression
from .parameters import sympy_evaluate


def _simplify_expression(boolean_expression: Expression, context: Z3SolverExpression) -> Optional[Expression]:
    if boolean_expression.type != bool:
        raise TypeError("Can only simplify booleans")
    if context.is_known_to_imply(boolean_expression):
        return basic_true
    if context.is_known_to_imply(~boolean_expression):
        return basic_false
    return None


def _collect_subset_expressions_helper(expression: Expression, test: Callable[[Expression], bool],
                                       result: Set[Expression]):
    if test(expression):
        result.add(expression)
    for sub_expression in expression.subexpressions:
        _collect_subset_expressions_helper(sub_expression, test, result)


def _collect_subset_expressions(expression: Expression,
                                test: Callable[[Expression], bool] = lambda _: True, ) -> Set[Expression]:
    """
    Collect all subexpressions including the ones that are not immediate subexpressions and self, with a filter
    `test`.
    """
    result = set()
    _collect_subset_expressions_helper(expression, test, result)
    return result


class ContextSimplifier(Simplifier):
    sympy_interpreter = SymPyInterpreter()

    @staticmethod
    def _simplify_pass(expression: Expression, context: Z3SolverExpression) -> Expression:
        try:
            result = ContextSimplifier.sympy_interpreter.simplify(expression, context)
        except ConversionError:
            result = expression

        # replace boolean expressions
        all_boolean_subexpressions = _collect_subset_expressions(result, lambda expr: expr.type == bool)
        for boolean_subexpression in all_boolean_subexpressions:
            simplified_subexpression = _simplify_expression(boolean_subexpression, context)
            if simplified_subexpression is not None:
                result = result.replace(boolean_subexpression, simplified_subexpression)
                if result is None:
                    raise

        # replace variables
        for variable, replacement in context.variable_replacement_dict.items():
            result = result.replace(variable, replacement)
        return result

    def simplify(self, expression: Expression, context: Z3SolverExpression) -> Expression:
        with sympy_evaluate(True):  # To work around a bug in SymPy (see context_simplifier_test.py/test_sympy_bug).
            if not isinstance(context, Z3SolverExpression):
                raise ValueError("ContextSimplifier expects a Z3SolverExpression context.")
            if not context.satisfiability_is_known:
                raise Context.UnknownError()
            if context.unsatisfiable:
                raise ValueError(f"Context {context} is unsatisfiable.")

            new_expression = ContextSimplifier._simplify_pass(expression, context)
            print(f'..={new_expression}')
            while not new_expression.syntactic_eq(expression):
                expression = new_expression
                new_expression = ContextSimplifier._simplify_pass(expression, context)
                print(f'new={new_expression}')
            return new_expression
