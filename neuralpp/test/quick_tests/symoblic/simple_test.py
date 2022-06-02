"""
Tests of features covered in the following tutorials of SymPy.
https://docs.sympy.org/latest/tutorial/intro.html
https://docs.sympy.org/latest/tutorial/gotchas.html
https://docs.sympy.org/latest/tutorial/basic_operations.html
"""
import math

from sympy import *


def test_symbols_basic() -> None:
    """SymPy supports simple expression conversions. e.g., `x + 2 * y` is the same as `y + x + y`"""
    x, y = symbols('x y')
    expr = x + 2 * y
    assert expr == x + 2 * y
    assert expr == y + x + y


def test_expand_and_factor() -> None:
    """
    The simplification SymPy automatically performs is very limited. For example,
    x(x+2y) is not "equal" in the sense of SymPy's symbolic representation as x^2+2xy.
    Some simple customized simplification operations are `expand` and `factor`. Shown in this test case.
    """
    x, y = symbols('x y')
    factor_form_expr = x * (x + 2 * y)
    expand_form_expr = x ** 2 + 2 * x * y
    # Equation `factor_form_expr == expand_form_expr` would pass Z3's checking as they are "equal" in the logical sense.
    assert factor_form_expr != expand_form_expr
    assert expand(factor_form_expr) == expand_form_expr
    assert factor_form_expr == factor(expand_form_expr)


def test_computation() -> None:
    """ Symbolic computation """
    x, t, z, nu = symbols('x t z nu')
    # for simple formulas integrate and diff are invertible
    formulas = (x, sin(x), cos(x), sin(x) * cos(x), exp(x), sin(x)*exp(x))
    for formula in formulas:
        assert integrate(diff(formula, x), x) == formula
        assert diff(integrate(formula, x), x) == formula
    # but this would fail:
    complex_formula = exp(x) * x + x
    assert diff(integrate(complex_formula, x), x) != complex_formula
    # since
    assert diff(integrate(complex_formula, x), x) == x + (x-1) * exp(x) + exp(x)
    # and
    assert complex_formula != x + (x-1) * exp(x) + exp(x)


def test_change_after_create() -> None:
    """changing the binding of python variable does not affect expression already created """
    x = symbols('x')
    expr = x + 1
    x = 2
    assert expr == symbols('x') + 1


def test_gotchas_xor_and_divide() -> None:
    """Some gotchas of the library: ^ is xor, / can be float division. """
    # ^ in SymPy is reserved for xor (not exponentiation), as is in Python.
    x, y = symbols('x y')
    assert x ^ y == Xor(x, y)
    assert x ^ y != x ** y

    # / in Python3 is float division. So SymPy uses Rational() explicitly.
    # the following tests would pass as the rational 1/2 equals the division result 1/2=0.5
    assert Rational(1, 2) == 1/2
    assert Rational(1, 2) == Integer(1)/Integer(2)
    # things are a little different when it comes to 1/3
    assert Rational(1, 3) != 1/3
    assert Rational(1, 3) == Integer(1)/Integer(3)


def test_substitution():
    """ Test variable substitution. """
    x, y, z = symbols("x y z")
    expr = cos(x) + 1
    assert expr.subs(x, y) == cos(y) + 1
    assert expr.subs(x, x**y) == cos(x**y) + 1


def test_eval():
    """ Test numerical evaluation, including a quantifier example. """
    x, i, k = symbols("x i k")
    assert sqrt(8) != math.sqrt(8)
    assert sqrt(8).evalf() == math.sqrt(8)
    assert cos(2*x).evalf(subs={x: 2.4}) == math.cos(2*2.4)

    expr = Sum(Indexed('x', i), (i, 0, 3))  # expr is a quantified expression
    expr_as_func = lambdify(x, expr)  # lambdify() turns the expression into a python function
    assert expr_as_func([1, 2, 3, 4, 5]) == 10

    # doit() can also do computation/simplification, note the interval can be infinite
    assert Sum(x**k/factorial(k), (k, 0, oo)).doit() == exp(x)
    # Doc in Sum() defines the following behavior which is a little weird
    assert Sum(k, (k, i, i - 100)).doit() == -Sum(k, (k, i - 99, i - 1)).doit()
