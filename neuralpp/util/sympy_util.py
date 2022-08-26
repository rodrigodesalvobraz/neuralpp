from functools import cache
from typing import List, Tuple

import sympy
from sympy.core.function import UndefinedFunction


def sympy_piecewise_from_if_then_else(
    if_: sympy.Basic, then_: sympy.Basic, else_: sympy.Basic
) -> sympy.Piecewise:
    """In Piecewise, conditional comes after clause."""
    return sympy.Piecewise((then_, if_), (else_, True))


def sympy_piecewise_to_if_then_else(
    piecewise: sympy.Piecewise,
) -> Tuple[sympy.Basic, sympy.Basic, sympy.Basic]:
    return piecewise.args[0][1], piecewise.args[0][0], piecewise.args[1][0]


def fold_sympy_piecewise(
    piecewise_args: List[Tuple[sympy.Basic, sympy.Basic]]
) -> sympy.Piecewise:
    """`fold` any sympy piecewise to 2-entry piecewise. E.g.,
    Piecewise((s0, c0), (s1, c1), (s2, True)) will be folded into
    Piecewise((s0, c0), (Piecewise((s1, c1), (s2, True)), True)
    """
    if len(piecewise_args) < 2:
        raise TypeError("piecewise is expected to have at least two entries")
    elif len(piecewise_args) == 2:
        return sympy.Piecewise(*piecewise_args)
    else:
        return sympy.Piecewise(
            piecewise_args[0], (fold_sympy_piecewise(piecewise_args[1:]), True)
        )

@cache
def get_sympy_integral(sympy_expression, x):
    return sympy.Integral(sympy_expression, x).doit()

def is_sympy_uninterpreted_function(expression: sympy.Basic) -> bool:
    return isinstance(expression, UndefinedFunction)


def is_sympy_value(sympy_object: sympy.Basic) -> bool:
    return isinstance(sympy_object, sympy.Number) or isinstance(
        sympy_object, sympy.logic.boolalg.BooleanAtom
    )


def is_sympy_sum(sympy_object: sympy.Basic) -> bool:
    return isinstance(sympy_object, sympy.Sum)


def is_sympy_integral(sympy_object: sympy.Basic) -> bool:
    return isinstance(sympy_object, sympy.Integral)
