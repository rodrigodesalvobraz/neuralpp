"""
Tests of SymPy's simplication functions.
https://docs.sympy.org/latest/tutorial/simplification.html
"""
from sympy import *


def test_simplify() -> None:
    """
    One can just use sympy.simplify(). It's push-button but has 2 caveats:
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


def test_polynomial_rational() -> None:
    """Test simplification for polynomial/rational."""
    x, y, z = symbols('x y z')
    # expand() expands.
    assert expand((x + 2)**2) == x**2 + 4*x + 4
    assert expand((x + 2)*(x - 2)) == x**2 - 4

    # factor() turns a polynomial into an irreducible product of factors.
    assert factor(x ** 3 - 1) == (x - 1)*(x**2 + x + 1)
    assert factor(x**2*z + 4*x*y*z + 4*y**2*z) == z*(x + 2*y)**2
    # here the definition of "polynomial" is liberal:
    assert factor(exp(x)**2 + 2*exp(x)*sin(x) + sin(x)**2) == (exp(x) + sin(x))**2

    # collect() collects common powers, i.e., \"sort\" a polynomial.
    assert collect(x**2 + y*x**2, x) == (1 + y)*x**2
    # it is useful to be used together with .coeff()
    assert collect(x**2 + y*x**2, x).coeff(x, 2) == 1 + y

    # cancel() cancels common factors between numerator and denominator.
    assert cancel((x**2 + 4*x + 4)/(x**2 + 2*x)) == (x + 2)/x
    # according to the doc, factor() is a superset of cancel(), but the latter is more efficient.
    assert factor((x**2 + 4*x + 4) / (x**2 + 2*x)) == (x + 2) / x


def test_trigonometric() -> None:
    """Test simplification for trigonometric functions."""
    x, y, z = symbols('x y z')
    assert trigsimp(sin(x)**2 + cos(x)**2) == 1
    # the following would fail if replace Integer(1)/Integer(2) with 1/2
    assert trigsimp(sin(x)**4 - 2*cos(x)**2*sin(x)**2 + cos(x)**4) == cos(4*x)/2 + Integer(1)/Integer(2)
    assert trigsimp(sin(x)**4 - 2*cos(x)**2*sin(x)**2 + cos(x)**4) != cos(4*x)/2 + 1/2
    assert expand_trig(sin(x + y)) == sin(x)*cos(y) + sin(y)*cos(x)


def test_powers() -> None:
    """Test simplification for powers."""
    x, y = symbols('x y', positive=True)
    a, b = symbols('a b', real=True)
    assert powsimp(x**a*y**a) == (x*y)**a
    # note that the above equation is not always true. e.g., x=y=-1, a=1/2
    xx, yy = symbols('xx yy')  # no specification of variables being positive
    assert powsimp(xx**a*yy**a) != (xx*yy)**a


def test_exp_and_log() -> None:
    """Test simplification for exponentials and logarithms."""
    x, y = symbols('x y', positive=True)
    n = symbols('n', real=True)
    assert expand_log(log(x*y)) == log(x) + log(y)
    assert logcombine(n*log(x)) == log(x**n)
