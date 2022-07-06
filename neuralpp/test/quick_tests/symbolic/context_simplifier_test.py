import pytest
import sympy
import z3
import operator
from neuralpp.symbolic.context_simplifier import ContextSimplifier
from neuralpp.symbolic.sympy_expression import SymPyVariable, sympy_Cond
from neuralpp.symbolic.z3_expression import Z3Variable, Z3SolverExpression
from neuralpp.symbolic.constants import if_then_else


def test_context_simplifier1():
    si = ContextSimplifier()
    sympy_x = SymPyVariable(sympy.symbols('x'), int)
    x = Z3Variable(z3.Int('x'))
    y = Z3Variable(z3.Int('y'))
    context = Z3SolverExpression()
    context = context & (x == y)  # parenthesis is necessary
    context = context & (y > 4)
    result = si.simplify(sympy_x > 3, context)
    assert result.value  # which means result is simplified to "True"


def test_context_simplifier2():
    si = ContextSimplifier()
    x = Z3Variable(z3.Int('x'))
    y = Z3Variable(z3.Int('y'))
    context = Z3SolverExpression()
    context = context & (x == y)
    context = context & (y > 4)
    expr = if_then_else(y < if_then_else(x < 3, 1, 2), x * y, x + y)
    result = si.simplify(expr, context)
    assert result.function.value == operator.add
    assert result.arguments[0].name == 'x'
    assert result.arguments[1].name == 'y'


def test_sympy_bug():
    x, y = sympy.symbols('x y')
    # We should be able to create the following expression in SymPy:
    #  if y < (if x<3 then 1 else 2) then x * y else x + y
    expr = sympy_Cond(y < sympy.Piecewise((1, x < 3), (2, True)), x * y, x + y)

    # But if we set sympy.core.parameters to False (by `with sympy.evaluate(False)`) and try to create the same
    # expression, SymPy will raise an error.
    with pytest.raises(Exception):  # non-deterministically TypeError/RecursiveError
        with sympy.evaluate(False):
            sympy_Cond(y < sympy.Piecewise((1, x < 3), (2, True)), x*y, x+y)
    # And this has nothing to do with our shorthand `sympy_Cond`.
    with pytest.raises(Exception):  # non-deterministically TypeError/RecursiveError
        with sympy.evaluate(False):
            sympy.Piecewise((x*y, y < sympy.Piecewise((1, x < 3), (2, True))), (x+y, True))

    # This means we cannot create such expressions in SymPy unless it is fixed in the library.
    # However, for simplify() methods, we don't need to stop SymPy from evaluating, so we can set evaluate=True
    # and work around this bug at least for simplify()'s cases.


def test_sympy_bug_detail():
    import sympy.functions.elementary.piecewise
    x = sympy.symbols('x')
    with pytest.raises(Exception):
        with sympy.evaluate(False):
            sympy_Cond(x < sympy.Piecewise((1, x < 3), (2, True)), 1, 2)

    # the minimum to reproduce the same bug:
    with pytest.raises(Exception):
        with sympy.evaluate(False):
            sympy.functions.elementary.piecewise.ExprCondPair(1, x < sympy.Piecewise((1, x < 3), (2, True)))
