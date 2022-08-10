import fractions

import pytest
import sympy
import z3
from typing import Callable

from neuralpp.symbolic.quantifier_free_normalizer import QuantifierFreeNormalizer
from neuralpp.symbolic.general_normalizer import GeneralNormalizer
from neuralpp.symbolic.basic_expression import BasicVariable, BasicSummation, BasicConstant, BasicIntegral
from neuralpp.symbolic.constants import if_then_else
from neuralpp.symbolic.z3_expression import Z3SolverExpression, Z3Variable
from neuralpp.symbolic.sympy_expression import SymPyExpression


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
    i_range = Z3SolverExpression.from_expression(0 < i) & (i < 10)
    sum_ = BasicSummation(int, i, i_range, i)

    context = empty_context
    normalizer = GeneralNormalizer()
    assert normalizer.normalize(sum_, context).syntactic_eq(BasicConstant(45))

    assert normalizer.normalize(BasicSummation(int, i, i_range, i + 1), context).syntactic_eq(BasicConstant(54))

    x = BasicVariable('x', int)
    i_range_symbolic = Z3SolverExpression.from_expression(x < i) & (i < 100)
    xx = sympy.symbols('x')
    assert normalizer.normalize(BasicSummation(int, i, i_range_symbolic, i + 1), context).sympy_object == \
           -xx ** 2 / 2 - 3 * xx / 2 + 5049

    context = Z3SolverExpression.from_expression(i < 5)
    # raises ValueError because the context should not contain index (in this case i),
    # since the index of the quantifier expression is not a free variable and not visible to the context.
    with pytest.raises(ValueError):
        normalizer.normalize(sum_, context)

    j = BasicVariable('j', int)
    sum_ = BasicSummation(int, i, Z3SolverExpression.from_expression(j < i) & (i < 10), i + j)
    context = Z3SolverExpression.from_expression(j == 5)
    # 6 + 7 + 8 + 9 + 5 * 4 = 50
    assert normalizer.normalize(sum_, context).syntactic_eq(BasicConstant(50))  # SymPy switched 5 and i

    context = Z3SolverExpression.from_expression(j == 10)
    assert normalizer.normalize(sum_, context).syntactic_eq(BasicConstant(0, int))

    k = BasicVariable('k', int)
    jj, kk = sympy.symbols('j k')
    sum_ = BasicSummation(int, i, Z3SolverExpression.from_expression(j < i) & (i < k), if_then_else(j > 5, i + j, i))
    assert normalizer.normalize(sum_, empty_context).syntactic_eq(
        if_then_else(j > 5,
                     SymPyExpression.from_sympy_object(
                         - jj ** 2 / 2 - jj * (jj - kk + 1) - jj / 2 + kk ** 2 / 2 - kk / 2,
                         {jj: int, kk: int}),
                     SymPyExpression.from_sympy_object(- jj ** 2 / 2 - jj / 2 + kk ** 2 / 2 - kk / 2,
                                                       {jj: int, kk: int}),
                     ))
    assert normalizer.normalize(sum_, Z3SolverExpression.from_expression(j == 6)).syntactic_eq(
        SymPyExpression.from_sympy_object(
            kk ** 2 / 2 + 11 * kk / 2 - 63,
            {kk: int})
    )

    sum_ = BasicSummation(int, i, Z3SolverExpression.from_expression(j < i), if_then_else(i > 5, i + j, i))
    assert normalizer.normalize(sum_, empty_context).syntactic_eq(
        BasicSummation(int, i, Z3SolverExpression.from_expression(j < i) & (5 < i), i + j) +
        BasicSummation(int, i, Z3SolverExpression.from_expression(j < i) & ~(5 < i), i))

    assert (normalizer.normalize(BasicSummation(int, j, Z3SolverExpression.from_expression(10 > j) & (j > 5), i + j),
                                 empty_context)).syntactic_eq(30 + 4 * i)
    # Normalization of nested quantifier expression
    sum_ = BasicSummation(int, j, Z3SolverExpression.from_expression(j < 10), if_then_else(j > 5, i + j, sum_))
    assert normalizer.normalize(sum_, empty_context).syntactic_eq(
        30 + 4 * i +
        BasicSummation(int, j, Z3SolverExpression.from_expression(10 > j) & ~(5 < j),
                       BasicSummation(int, i, Z3SolverExpression.from_expression(j < i) & (5 < i), i + j) +
                       BasicSummation(int, i, Z3SolverExpression.from_expression(j < i) & ~(5 < i), i)))

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
    expr = f(if_then_else(A, B, C), BasicSummation(int, D, Z3SolverExpression.from_expression(0 < D) & (D < 10),
                                                   B * if_then_else(B > 4,
                                                                    B + C,
                                                                    BasicSummation(int, C,
                                                                                   Z3SolverExpression.from_expression(0 < C) & (C < 10),
                                                                                   if_then_else(B < 5, C, 1)))))
    BB, CC = sympy.symbols('B C')
    product = SymPyExpression.from_sympy_object(9 * BB * (BB + CC), {BB: int, CC: int})
    assert normalizer.normalize(expr, empty_context).syntactic_eq(
        if_then_else(A,
                     if_then_else(B > 4,
                                  f(B, product),
                                  f(B, 405 * B)),
                     if_then_else(B > 4,
                                  f(C, product),
                                  f(C, 405 * B))))
    assert normalizer.normalize(expr, Z3SolverExpression.from_expression(A) & (B > 4)).syntactic_eq(f(B, product))

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

    expr = f(if_then_else(A, B, C), BasicSummation(int, D, Z3SolverExpression.from_expression(0 < D) & (D < 10),
                                                   B * if_then_else(B == 4, B + C, 1)))
    product2 = SymPyExpression.from_sympy_object(144 + 36 * CC, {CC: int})
    assert normalizer.normalize(expr, empty_context).syntactic_eq(
        if_then_else(A,
                     if_then_else(B == 4, f(4, product2), f(B, 9 * B)),
                     if_then_else(B == 4, f(C, product2), f(C, 9 * B))))


def test_quantifier_normalizer_integration():
    i = BasicVariable('i', int)
    empty_context = Z3SolverExpression()
    i_range = Z3SolverExpression.from_expression(0 < i) & (i < 10)
    integral = BasicIntegral(i, i_range, i)

    context = empty_context
    normalizer = GeneralNormalizer()
    # 1/2 * i ** 2
    assert normalizer.normalize(integral, context).syntactic_eq(BasicConstant(50))
    # 1/2 * i ** 2 + i
    assert normalizer.normalize(BasicIntegral(i, i_range, i + 1), context).syntactic_eq(BasicConstant(60))

    #          *
    #       /    \
    #     if A   Integral
    #     /  \    |
    #    B    C  B * if B>4
    #                /    \
    #               B+C   Integral
    #                       |
    #                      if B<5
    #                      / \
    #                     C   1
    # should be normalized to
    #              if A
    #           /        \
    #      if B>4       if B>4
    #       /    \        /   \
    #      *      ..   ..      * (= C * 10 * 50 = 500 * B * C)
    #     /\                  /  \
    #    B  Integral(D)      C   Integral(D)    (= *10)
    #        |                    |
    #       B*(B+C)             B*Integral(C)  (= B * 1/2 * C ** 2 {C:[0,10]} = 50 * B)
    f = BasicVariable('f', Callable[[int, int], int])
    A = BasicVariable('A', bool)
    B = BasicVariable('B', int)
    C = BasicVariable('C', int)
    D = BasicVariable('D', int)
    expr = if_then_else(A, B, C) * BasicIntegral(D, Z3SolverExpression.from_expression(0 < D) & (D < 10),
                                                 B * if_then_else(B > 4,
                                                                  B + C,
                                                                  BasicIntegral(C,
                                                                                Z3SolverExpression.from_expression(0 < C) & (C < 10),
                                                                                if_then_else(B < 5, C, 1))))
    BB, CC = sympy.symbols('B C')
    product = SymPyExpression.from_sympy_object(10 * BB ** 2 * (BB + CC), {BB: int, CC: int})
    product2 = SymPyExpression.from_sympy_object(10 * BB * CC * (BB + CC), {BB: int, CC: int})
    product3 = SymPyExpression.from_sympy_object(500 * BB * CC, {BB: int, CC: int})
    assert normalizer.normalize(expr, empty_context).syntactic_eq(
        if_then_else(A,
                     if_then_else(B > 4, product, 500 * B ** 2),
                     if_then_else(B > 4, product2, product3),
                     ))
    assert normalizer.normalize(expr, Z3SolverExpression.from_expression(A) & (B > 4)).syntactic_eq(product)
