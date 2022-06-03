"""
Test of Z3Py.
"""
from z3 import *
from typing import Optional


def is_valid(predicate: ExprRef) -> bool:
    """
    If we want to ask an SMT solver to check if a predicate PRED is valid (i.e., always true), we need to
    ask in a not very intuitive way: "is `not PRED` unsatisfiable?"
    If `not PRED` is not unsat, then PRED is NOT always true, since there must exist a counter-example that satisfies
    `not PRED`, which we can get by calling s.model().
    If `not PRED` is unsat, then PRED is always true.
    """
    s = Solver()
    s.add(Not(predicate))
    return s.check() == unsat


def test_is_valid() -> None:
    """ Test that is_valid() is implemented correctly. """
    x = Int('x')
    assert is_valid(Or(x > 0, x <= 0))
    assert is_valid(Implies(x > 1, x > 0))
    assert not is_valid(x > 0)
    assert not is_valid(Implies(x > 0, x > 1))


def test_z3_simplification() -> None:
    """ Test of z3's simplify() function. """
    x, y = Ints('x y')

    # simplify() can perform some trivial simplification.
    # == and != are overloaded. So we just use repr().
    assert repr(simplify(x < y + x + 2)) == "Not(y <= -2)"

    # We start from two equivalent condition. Ideally we want to derive cond2 automatically from cond1.
    cond1 = And(x > 2, x < 4)
    cond2 = x == 3
    # However, Z3 cannot simplify 2 < x < 4 into x == 3
    assert repr(simplify(cond1)) == "And(Not(x <= 2), Not(4 <= x))"
    # But we can always ask z3 to solve the problem "does 2 < x < 4 imply x == 3?"
    assert is_valid(Implies(cond1, cond2))


def test_z3_solve_nonlinear_polynomial() -> None:
    """
    Z3 can solve nonlinear polynomial constraints. In terms of expression power, it is a subset SymPy which
    also supports powers, exp/log and trigonometric.
    """
    x, y = Reals('x y')
    s = Solver()
    s.add(x ** 2 + y ** 2 > 3, x ** 3 + y < 5)
    assert s.check() == sat
    # call s.model() to get a solution


# def test_bitvec() -> None:
#
#
# def test_function() -> None:
#
#
# def test_quantifier() -> None:
