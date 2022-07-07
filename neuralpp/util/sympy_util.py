import sympy
from sympy.core.function import UndefinedFunction


def is_sympy_uninterpreted_function(expression: sympy.Basic) -> bool:
    return isinstance(expression, UndefinedFunction)
