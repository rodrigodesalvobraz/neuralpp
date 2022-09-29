"""
Test cases comparing Z3 and SymPy solver.
"""
import pytest
import sympy.core.power

from sympy import symbols, Poly, PolynomialError, Interval, oo
from sympy.solvers.inequalities import solve_rational_inequalities
from typing import List, Tuple
from z3 import Real, Solver, Ints, sat


def solve_inequalities_sympy(
    ineq_list: List[Tuple[sympy.core.Expr, sympy.core.Expr, str]]
):
    """
    SymPy's support for inequalities are a little hard to read. This is just a wrapper for inequalities.
    `ineq_list` is expected to be a list of (LHS, VARS, OP) where the inequality is "LHS OP 0", VARS contains all
    variables used in the inequality. So for example, (x + 1, x, >) represents "x + 1 > 0"
    """
    inequalities = []
    for lhs, variables, op in ineq_list:
        inequalities.append(((Poly(lhs), Poly(1, variables)), op))
    return solve_rational_inequalities([inequalities])


def test_compare_z3_and_sympy_solver():
    """
    A very simple inequality case which both Z3 and SymPy can solve. Note that "solve" in Z3's context means
    differently from that in SymPy's. The latter is trying to solve a harder problem.
    Z3 only finds one concrete solution if it is satisfiable, while SymPy's inequality solver tries to find an interval.
    """
    # z3 solve
    x = Real("x")
    s = Solver()
    s.add(x * x > 1, x < 0)  # x*x > 1 AND x < 0
    assert s.check() == sat
    # the type is int: just one solution, not a range. model() returns an example if check() == sat.
    x0: float = float(s.model()[x].as_decimal(prec=3))
    assert x0 < -1

    # sympy solve
    x = symbols("x")
    # The following just means # x * x > 1 and x < 0, same as the inequalities above.
    assert solve_inequalities_sympy(
        [(x**2 - 1, x, ">"), (x, x, "<")]
    ) == Interval.open(-oo, -1)


def test_compare_z3_and_sympy_solver_sympy_fail():
    """
    SymPy's solve() targets a harder goal. So the scope of its solvable problems are smaller.
    It fails some simple tasks solvable in Z3. For example, it doesn't support inequalities with multiple variables.
    """
    x, y = Ints("x y")
    s = Solver()
    s.add(x > 2, y < 10, x + 2 * y == 7)
    assert s.check() == sat
    x0: int = s.model()[x].as_long()
    y0: int = s.model()[y].as_long()
    assert x0 > 2 and y0 < 10 and x0 + 2 * y0 == 7

    # SymPy's solver is ill-equipped to solve the set of inequalities.
    x, y = symbols("x y")
    with pytest.raises(PolynomialError) as exc_info:
        solve_inequalities_sympy(
            [
                (x**2 - 1, x, ">"),
                (y - 10, y, "<"),
                (x + 2 * y - 7, (x, y), "=="),
            ]
        )
    assert "only univariate polynomials are allowed" in repr(exc_info)
