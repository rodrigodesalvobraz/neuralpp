import z3

from neuralpp.symbolic.normalizer import Normalizer
from neuralpp.symbolic.basic_expression import BasicVariable
from neuralpp.symbolic.constants import if_then_else
from neuralpp.symbolic.z3_expression import Z3SolverExpression, Z3Variable


def test_normalizer1():
    """
    if a > b then a + b else 3
    """
    normalizer = Normalizer()
    a = BasicVariable('a', int)
    b = BasicVariable('b', int)
    expr = if_then_else(a > b, a + b, 3)
    context = Z3SolverExpression()
    result1 = normalizer.normalize(expr, context)
    assert result1.structure_eq(expr)

    context = context & (a == b)
    result2 = normalizer.normalize(expr, context)
    assert result2.value == 3


def test_normalizer2():
    """
    f: bool -> bool -> int -> int
    f(a > b, b > c, 5)
    """
    normalizer = Normalizer()
    f = Z3Variable(z3.Function('f', z3.BoolSort(), z3.BoolSort(), z3.IntSort(), z3.IntSort()))
    a = BasicVariable('a', int)
    b = BasicVariable('b', int)
    c = BasicVariable('c', int)

    expr = f(a < b, b > c, 5)
    context = Z3SolverExpression()
    context = context & (f(True, True, 5) == 42) & (f(True, False, 5) == 99)
    assert isinstance(context, Z3SolverExpression)
    result = normalizer.normalize(expr, context)
    assert result.structure_eq(if_then_else(a < b,
                                            if_then_else(b > c, 42, 99),
                                            if_then_else(b > c, f(False, True, 5), f(False, False, 5))))


def test_normalizer3():
    """
    f: bool -> int -> int
    f(c, f((a > b) | c, 3))
    """
    normalizer = Normalizer()
    f = Z3Variable(z3.Function('f', z3.BoolSort(), z3.IntSort(), z3.IntSort()))
    a = BasicVariable('a', int)
    b = BasicVariable('b', int)
    c = BasicVariable('c', bool)

    expr = f(c, f((a > b) | c, 3))
    context = Z3SolverExpression()
    context = context & (f(True, 66) == 45) & (f(True, 3) == 66) & (f(False, 3) == 3)
    assert isinstance(context, Z3SolverExpression)
    result = normalizer.normalize(expr, context)
    assert result.structure_eq(if_then_else(c, 45, if_then_else(a > b, f(False, 66), 3)))
