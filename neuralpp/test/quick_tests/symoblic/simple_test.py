"""
Tests of features covered in the following tutorials of SymPy.
https://docs.sympy.org/latest/tutorial/intro.html
https://docs.sympy.org/latest/tutorial/gotchas.html
https://docs.sympy.org/latest/tutorial/basic_operations.html
"""
import math

# cannot use `from sympy import *` because that would import all the "test_*" functions in sympy,
# causing pytest to run them as well.
from sympy import symbols, expand, factor, sin, cos, diff, integrate, exp, Integer, Xor, Rational, \
    sqrt, Sum, lambdify, Indexed, factorial, oo


def test_symbols_basic():
    """SymPy supports simple expression conversions. e.g., `x + 2 * y` is the same as `y + x + y`"""
    x, y = symbols('x y')
    expr = x + 2 * y
    assert expr == x + 2 * y
    assert expr == y + x + y


def test_expand_and_factor():
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


def test_computation():
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


def test_change_after_create():
    """changing the binding of python variable does not affect expression already created """
    x = symbols('x')
    expr = x + 1
    x = 2
    assert expr == symbols('x') + 1


def test_gotchas_xor_and_divide():
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

    # lambdify() turns the expression into a python function. It can also be used to evaluate.
    add_expr = x + k
    add_func = lambdify((x, k), add_expr)
    assert add_func(1, 2) == 3
    assert add_func(3, 4) == 7
    assert add_func(5, -1) == (lambda a, b: a + b)(5, -1)

    expr = Sum(Indexed('x', i), (i, 0, 3))  # expr is a quantified expression
    expr_as_func = lambdify(x, expr)
    assert expr_as_func([1, 2, 3, 4, 5]) == 10


def test_sum():
    """ A more detailed test of the Sum quantifier.
    In quote of Sum's doc, the first argument is "the general form of terms in the series" and
    the second argument is "(dummy_variable, start, end), with dummy_variable taking all
    integer values from start through end. In accordance with long-standing mathematical convention,
    the end term is included in the summation."
    """
    x, i, k = symbols("x i k")
    # In the following example, `i` is not used in the series but just to indicate the index of 'x'.
    expr = Sum(Indexed('x', i), (i, 0, 100))
    expr_as_func = lambdify(x, expr)
    assert expr_as_func([j for j in range(101)]) == 5050

    # `i` can also be used in the series. An alternative way to the above example is the following.
    expr = Sum(i, (i, 0, 100))  # expr is a quantified expression
    # Here doit() does evaluation (e.g., it "does" the sum), note the interval can be infinite
    assert expr.doit() == 5050

    # Note doit() does *symbolic* evaluation. In the above example we get an integer only because there's no symbol.
    assert Sum(x**k/factorial(k), (k, 0, oo)).doit() == exp(x)
    # Doc in Sum() defines the following behavior which is a little weird.
    # If start > end in (i, start, end), it is defined to be the same as (i, end+1, start-1)
    # https://docs.sympy.org/latest/modules/concrete.html?highlight=sum#sympy.concrete.summations.Sum
    assert Sum(k, (k, i, i - 100)).doit() == -Sum(k, (k, i - 99, i - 1)).doit()
