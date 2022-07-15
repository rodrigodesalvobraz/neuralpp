import pytest
import z3

from neuralpp.symbolic.normalizer import Normalizer
from neuralpp.symbolic.basic_expression import BasicVariable, BasicSummation, BasicConstant
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
    f(a < b, b > c, 5)
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
    f(c, f((a > b) | c, 3)), context: f(True,3)==66, f(False,3)==3, f(True,66)==45

    The normalization process:
    f(c, f((a>b) | c, 3))
        --(split on c)-->
    if c then f(True, f(True,3)) else f(False, f(a>b,3))
        --(simplify by f(True, 3) == 66)-->
    if c then f(True, 66) else f(False, f(a>b,3))
        --(simplify by f(True, 66) == 45)-->
    if c then 45 else f(False, f(a>b,3))
        --(split on a>b)-->
    if c then 45 else if a>b then f(False, f(True,3)) else f(False, f(False,3))
        --(simplify by f(True, 3) == 66)-->
    if c then 45 else if a>b then f(False, 66) else f(False, f(False,3))
        --(simplify by f(False, 3) == 3)-->
    if c then 45 else if a>b then f(False, 66) else f(False, 3)
        --(simplify by f(False, 3) == 3)-->
    if c then 45 else if a>b then f(False, 66) else 3
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


def test_quantifier_normalizer():
    from neuralpp.symbolic.constants import int_add, int_multiply
    i = BasicVariable('i', int)
    empty_context = Z3SolverExpression()
    i_range = empty_context & (0 < i) & (i < 10)
    sum_ = BasicSummation(int, i, i_range, i)

    context = empty_context
    normalizer = Normalizer()
    assert normalizer.normalize(sum_, context).structure_eq(sum_)

    context = empty_context & (i < 5)
    with pytest.raises(ValueError):
        normalizer.normalize(sum_, context)

    j = BasicVariable('j', int)
    sum_ = BasicSummation(int, i, empty_context & (j < i) & (i < 10), i + j)
    context = empty_context & (j == 5)
    assert normalizer.normalize(sum_, context).structure_eq(
        BasicSummation(int, i, empty_context & (j < i) & (i < 10), 5 + i))  # SymPy switched 5 and i

    context = empty_context & (j == 10)
    assert normalizer.normalize(sum_, context).structure_eq(
        BasicSummation(int, i, empty_context & (j < i) & (i < 10), 10 + i))
    # not reduced to 0 because j < i < 10 is still satisfiable
    assert not normalizer.normalize(sum_, context).structure_eq(BasicConstant(0, int))

    sum_ = BasicSummation(int, i, empty_context & (j < i), if_then_else(j > 5, i + j, i))
    assert normalizer.normalize(sum_, empty_context).structure_eq(
        if_then_else(j > 5,
                     BasicSummation(int, i, empty_context & (j < i), i + j),
                     BasicSummation(int, i, empty_context & (j < i), i)))
    assert normalizer.normalize(sum_, empty_context & (j == 6)).structure_eq(
        BasicSummation(int, i, empty_context & (j < i), 6 + i))

    sum_ = BasicSummation(int, i, empty_context & (j < i), if_then_else(i > 5, i + j, i))
    assert normalizer.normalize(sum_, empty_context).structure_eq(
                     BasicSummation(int, i, empty_context & (j < i) & (i > 5), i + j) +
                     BasicSummation(int, i, empty_context & (j < i) & (i <= 5), i))

    # Normalization of nested quantifier expression is not supported, yet.
    sum_ = BasicSummation(int, j, empty_context & (j < 10), if_then_else(j > 5, i + j, sum_))
    # because SymPyQuantifierExpression is not implemented.
    with pytest.raises(NotImplementedError):
        normalizer.normalize(sum_, empty_context)