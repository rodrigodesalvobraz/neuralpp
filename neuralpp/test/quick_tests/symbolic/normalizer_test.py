import pytest
import z3
from typing import Callable

from neuralpp.symbolic.quantifier_free_normalizer import QuantifierFreeNormalizer
from neuralpp.symbolic.general_normalizer import GeneralNormalizer
from neuralpp.symbolic.basic_expression import BasicVariable, BasicSummation, BasicConstant, BasicFunctionApplication
from neuralpp.symbolic.constants import if_then_else
from neuralpp.symbolic.z3_expression import Z3SolverExpression, Z3Variable


def test_normalizer1():
    """
    if a > b then a + b else 3
    """
    normalizer = QuantifierFreeNormalizer()
    a = BasicVariable('a', int)
    b = BasicVariable('b', int)
    expr = if_then_else(a > b, a + b, 3)
    context = Z3SolverExpression()
    result1 = normalizer.normalize(expr, context)
    assert result1.syntactic_eq(expr)

    context = context & (a == b)
    result2 = normalizer.normalize(expr, context)
    assert result2.value == 3


def test_normalizer2():
    """
    f: bool -> bool -> int -> int
    f(a < b, b > c, 5)
    """
    normalizer = QuantifierFreeNormalizer()
    f = Z3Variable(z3.Function('f', z3.BoolSort(), z3.BoolSort(), z3.IntSort(), z3.IntSort()))
    a = BasicVariable('a', int)
    b = BasicVariable('b', int)
    c = BasicVariable('c', int)

    expr = f(a < b, b > c, 5)
    context = Z3SolverExpression()
    context = context & (f(True, True, 5) == 42) & (f(True, False, 5) == 99)
    assert isinstance(context, Z3SolverExpression)
    result = normalizer.normalize(expr, context)
    assert result.syntactic_eq(if_then_else(a < b,
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
    normalizer = QuantifierFreeNormalizer()
    f = Z3Variable(z3.Function('f', z3.BoolSort(), z3.IntSort(), z3.IntSort()))
    a = BasicVariable('a', int)
    b = BasicVariable('b', int)
    c = BasicVariable('c', bool)

    expr = f(c, f((a > b) | c, 3))
    context = Z3SolverExpression()
    context = context & (f(True, 66) == 45) & (f(True, 3) == 66) & (f(False, 3) == 3)
    assert isinstance(context, Z3SolverExpression)
    result = normalizer.normalize(expr, context)
    assert result.syntactic_eq(if_then_else(c, 45, if_then_else(a > b, f(False, 66), 3)))


def test_quantifier_normalizer():
    from neuralpp.symbolic.constants import int_add, int_multiply
    i = BasicVariable('i', int)
    empty_context = Z3SolverExpression()
    i_range = empty_context & (0 < i) & (i < 10)
    sum_ = BasicSummation(int, i, i_range, i)

    context = empty_context
    normalizer = GeneralNormalizer()
    assert normalizer.normalize(sum_, context).syntactic_eq(sum_)

    context = empty_context & (i < 5)
    # raises ValueError because the context should not contain index (in this case i),
    # since the index of the quantifier expression is not a free variable and not visible to the context.
    with pytest.raises(ValueError):
        normalizer.normalize(sum_, context)

    j = BasicVariable('j', int)
    sum_ = BasicSummation(int, i, empty_context & (j < i) & (i < 10), i + j)
    context = empty_context & (j == 5)
    assert normalizer.normalize(sum_, context).syntactic_eq(
        BasicSummation(int, i, empty_context & (j < i) & (i < 10), 5 + i))  # SymPy switched 5 and i

    context = empty_context & (j == 10)
    assert normalizer.normalize(sum_, context).syntactic_eq(BasicConstant(0, int))

    sum_ = BasicSummation(int, i, empty_context & (j < i), if_then_else(j > 5, i + j, i))
    assert normalizer.normalize(sum_, empty_context).syntactic_eq(
        if_then_else(j > 5,
                     BasicSummation(int, i, empty_context & (j < i), i + j),
                     BasicSummation(int, i, empty_context & (j < i), i)))
    assert normalizer.normalize(sum_, empty_context & (j == 6)).syntactic_eq(
        BasicSummation(int, i, empty_context & (j < i), 6 + i))

    sum_ = BasicSummation(int, i, empty_context & (j < i), if_then_else(i > 5, i + j, i))
    assert normalizer.normalize(sum_, empty_context).syntactic_eq(
        BasicSummation(int, i, empty_context & (j < i) & (5 < i), i + j) +
        BasicSummation(int, i, empty_context & (j < i) & ~(5 < i), i))

    sum1 = if_then_else(j > 5, i + j, sum_)

    # Normalization of nested quantifier expression
    sum_ = BasicSummation(int, j, empty_context & (j < 10), if_then_else(j > 5, i + j, sum_))
    assert normalizer.normalize(sum_, empty_context).syntactic_eq(
        BasicSummation(int, j, empty_context & (10 > j) & (j > 5), i + j) +
        BasicSummation(int, j, empty_context & (10 > j) & ~(5 < j),
                       BasicSummation(int, i, empty_context & (j < i) & (5 < i), i + j) +
                       BasicSummation(int, i, empty_context & (j < i) & ~(5 < i), i)))

    #          f
    #       /    \
    #     if A   Sum
    #     /  \    |
    #    B    C  B * if B>4
    #                /    \
    #               B+C    Sum
    #                       |
    #                      if B<5
    #                      / \
    #                     C   1
    # should be normalized to
    #              if A
    #           /        \
    #      if B>4       if B>4
    #       /    \        /   \
    #      f      ..   ..      f
    #     /\                  /  \
    #    B  Sum              C   Sum
    #        |                    |
    #       B*(B+C)             B*Sum(C)
    f = BasicVariable('f', Callable[[int, int], int])
    A = BasicVariable('A', bool)
    B = BasicVariable('B', int)
    C = BasicVariable('C', int)
    D = BasicVariable('D', int)
    expr = f(if_then_else(A, B, C), BasicSummation(int, D, empty_context & (0 < D) & (D < 10),
                                                   B * if_then_else(B > 4,
                                                                    B + C,
                                                                    BasicSummation(int, C,
                                                                                   empty_context & (0 < C) & (C < 10),
                                                                                   if_then_else(B < 5, C, 1)))))
    assert normalizer.normalize(expr, empty_context).syntactic_eq(
        if_then_else(A,
                     if_then_else(B > 4,
                                  f(B, BasicSummation(int, D, empty_context & (0 < D) & (D < 10), B * (B + C))),
                                  f(B, BasicSummation(int, D, empty_context & (0 < D) & (D < 10),
                                                      B * (
                                                          BasicSummation(int, C, empty_context & (0 < C) & (C < 10), C))
                                                      ))),
                     if_then_else(B > 4,
                                  f(C, BasicSummation(int, D, empty_context & (0 < D) & (D < 10), B * (B + C))),
                                  f(C, BasicSummation(int, D, empty_context & (0 < D) & (D < 10),
                                                      B * (
                                                          BasicSummation(int, C, empty_context & (0 < C) & (C < 10), C))
                                                      )))))
    assert normalizer.normalize(expr, empty_context & A & (B > 4)).syntactic_eq(
        f(B, BasicSummation(int, D, empty_context & (0 < D) & (D < 10), B * (B + C))))

    # if we have "B==4" in the context of a subtree, B will be substituted by 4
    #          f
    #       /    \
    #     if A   Sum
    #     /  \    |
    #    B    C  B * if B==4
    #                /    \
    #               B+C    1
    # should be normalized to
    #              if A
    #           /        \
    #      if B==4       if B==4
    #       /    \        /   \
    #      f      ..   ..      f
    #     /\                  /  \
    #    4  Sum              C   Sum
    #        |                    |
    #       16+4*C                B

    expr = f(if_then_else(A, B, C), BasicSummation(int, D, empty_context & (0 < D) & (D < 10),
                                                   B * if_then_else(B == 4, B + C, 1)))
    assert normalizer.normalize(expr, empty_context).syntactic_eq(
        if_then_else(A,
                     if_then_else(B == 4,
                                  f(B, BasicSummation(int, D, empty_context & (0 < D) & (D < 10), 16 + 4 * C)),
                                  f(B, BasicSummation(int, D, empty_context & (0 < D) & (D < 10), B))),
                     if_then_else(B == 4,
                                  f(C, BasicSummation(int, D, empty_context & (0 < D) & (D < 10), 16 + 4 * C)),
                                  f(C, BasicSummation(int, D, empty_context & (0 < D) & (D < 10), B)))))
