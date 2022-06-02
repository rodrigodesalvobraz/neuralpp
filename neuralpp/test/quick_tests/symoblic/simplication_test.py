"""
Tests of SymPy's simplication functions.
https://docs.sympy.org/latest/tutorial/simplification.html
"""
from sympy import *


def test_simplify() -> None:
    """
    One can just use sympy.simplify().
    It's push-button but has 2 caveats:
    1. "simplest" form is not well-defined. (showcased by last set of asserts)
    2. it can be unnecessarily slow due to generality.
    """
    x = symbols('x')
    assert simplify(sin(x)**2 + cos(x)**2) == 1
    assert simplify((x**3 + x**2 - x - 1)/(x**2 + 2*x + 1)) == x - 1
    assert simplify(gamma(x) / gamma(x - 2)) == (x - 2) * (x - 1)
    # we may want to simplify (x**2 + 2*x + 1) into (x+1)**2 but that's not "simplest" recognized by simplify()
    assert simplify(x**2 + 2*x + 1) != (x + 1)**2
    # instead, we can call more specific simplification function.
    assert factor(x**2 + 2*x + 1) == (x + 1)**2


def test_expand() -> None:
    """ The function expand() expands. """
    x = symbols('x')
    assert expand((x + 2)**2) == x**2 + 4*x + 4
    assert expand((x + 2)*(x - 2)) == x**2 - 4


def test_factor() -> None:
    """ The function factor() turns a polynomial into an irreducible product of factors. """
    x, y, z = symbols('x y z')
    assert factor(x ** 3 - 1) == (x - 1)*(x**2 + x + 1)
    assert factor(x**2*z + 4*x*y*z + 4*y**2*z) == z*(x + 2*y)**2
    # here the definition of "polynomial" is liberal:
    assert factor(exp(x)**2 + 2*exp(x)*sin(x) + sin(x)**2) == (exp(x) + sin(x))**2


def test_collect() -> None:
    """The function collect() collects common powers, i.e., "sort" a polynomial. """
    x, y = symbols('x y')
    assert collect(x**2 + y*x**2, x) == (1 + y)*x**2
    # it is useful to be used together with .coeff()
    assert collect(x**2 + y*x**2, x).coeff(x, 2) == 1 + y


def test_cancel() -> None:
    """The function cancel() cancels common factors between numerator and denominator."""
    x, y, z = symbols('x y z')
    assert cancel((x**2 + 4*x + 4)/(x**2 + 2*x)) == (x + 2)/x
    # according to the doc, factor() is a superset of cancel() but the latter is more efficient.
    assert factor((x ** 2 + 4 * x + 4) / (x ** 2 + 2 * x)) == (x + 2) / x
