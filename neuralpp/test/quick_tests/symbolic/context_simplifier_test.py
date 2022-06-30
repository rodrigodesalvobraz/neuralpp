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
    with pytest.raises(Exception):  # non-deterministically TypeError/RecursiveError
        with sympy.evaluate(False):
            sympy_Cond(y < sympy.Piecewise((1, x < 3), (2, True)), x*y, x+y)
    with pytest.raises(Exception):  # non-deterministically TypeError/RecursiveError
        with sympy.evaluate(False):
            sympy.Piecewise((x*y, y < sympy.Piecewise((1, x < 3), (2, True))), (x+y, True))

    with sympy.evaluate(True):
        sympy_Cond(y < sympy.Piecewise((1, x < 3), (2, True)), x*y, x+y)
        sympy.Piecewise((x * y, y < sympy.Piecewise((1, x < 3), (2, True))), (x + y, True))

