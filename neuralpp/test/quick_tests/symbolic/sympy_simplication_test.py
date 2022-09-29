"""
Tests of SymPy's simplication functions.
https://docs.sympy.org/latest/tutorial/simplification.html
"""
import pytest
from sympy import (
    symbols,
    expand,
    factor,
    sin,
    cos,
    simplify,
    exp,
    Integer,
    gamma,
    collect,
    trigsimp,
    expand_trig,
    powsimp,
    expand_log,
    log,
    logcombine,
    cancel,
)
import sympy


def test_simplify():
    """
    One can just use sympy.simplify(). It's general but has 2 caveats:
    1. "simplest" form is not well-defined. (showcased by last set of asserts)
    2. it can be unnecessarily slow due to generality.
    """
    x = symbols("x")
    one = sin(x) ** 2 + cos(x) ** 2
    assert simplify(one) == 1
    assert simplify(sin(x) ** 2 + cos(x) ** 2) == 1
    assert simplify((x**3 + x**2 - x - 1) / (x**2 + 2 * x + 1)) == x - 1
    assert simplify(gamma(x) / gamma(x - 2)) == (x - 2) * (x - 1)
    # we may want to simplify (x**2 + 2*x + 1) into (x+1)**2 but that's not "simplest" recognized by simplify()
    assert simplify(x**2 + 2 * x + 1) != (x + 1) ** 2
    # instead, we can call more specific simplification function.
    assert factor(x**2 + 2 * x + 1) == (x + 1) ** 2


def test_polynomial_and_rational_simplification():
    """Test simplification for polynomial/rational."""
    x, y, z = symbols("x y z")
    # expand() expands.
    assert expand((x + 2) ** 2) == x**2 + 4 * x + 4
    assert expand((x + 2) * (x - 2)) == x**2 - 4

    # factor() turns a polynomial into an irreducible product of factors.
    assert factor(x**3 - 1) == (x - 1) * (x**2 + x + 1)
    assert (
        factor(x**2 * z + 4 * x * y * z + 4 * y**2 * z)
        == z * (x + 2 * y) ** 2
    )
    # here the definition of "polynomial" is liberal:
    assert (
        factor(exp(x) ** 2 + 2 * exp(x) * sin(x) + sin(x) ** 2)
        == (exp(x) + sin(x)) ** 2
    )

    # collect() collects common powers, i.e., \"sort\" a polynomial.
    assert collect(x**2 + y * x**2, x) == (1 + y) * x**2
    # it is useful to be used together with .coeff()
    assert collect(x**2 + y * x**2, x).coeff(x, 2) == 1 + y

    # cancel() cancels common factors between numerator and denominator.
    assert cancel((x**2 + 4 * x + 4) / (x**2 + 2 * x)) == (x + 2) / x
    # according to the doc, factor() is a superset of cancel(), but the latter is more efficient.
    assert factor((x**2 + 4 * x + 4) / (x**2 + 2 * x)) == (x + 2) / x


def test_trigonometric_simplification():
    """Test simplification for trigonometric functions."""
    x, y, z = symbols("x y z")
    assert trigsimp(sin(x) ** 2 + cos(x) ** 2) == 1
    # the following would fail if replace Integer(1)/Integer(2) with 1/2
    assert trigsimp(
        sin(x) ** 4 - 2 * cos(x) ** 2 * sin(x) ** 2 + cos(x) ** 4
    ) == cos(4 * x) / 2 + Integer(1) / Integer(2)
    assert (
        trigsimp(sin(x) ** 4 - 2 * cos(x) ** 2 * sin(x) ** 2 + cos(x) ** 4)
        != cos(4 * x) / 2 + 1 / 2
    )
    assert expand_trig(sin(x + y)) == sin(x) * cos(y) + sin(y) * cos(x)


def test_powers_simplification():
    """Test simplification for powers."""
    x, y = symbols("x y", positive=True)
    a, b = symbols("a b", real=True)
    assert powsimp(x**a * y**a) == (x * y) ** a
    # note that the above equation is not always true. e.g., x=y=-1, a=1/2
    xx, yy = symbols("xx yy")  # no specification of variables being positive
    assert powsimp(xx**a * yy**a) != (xx * yy) ** a


def test_exp_and_log_simplification():
    """Test simplification for exponentials and logarithms."""
    x, y = symbols("x y", positive=True)
    n = symbols("n", real=True)
    assert expand_log(log(x * y)) == log(x) + log(y)
    assert logcombine(n * log(x)) == log(x**n)


def test_unevaluate():
    """
    In creating our symbolic representation, we want SymPy to not automatically evaluate or simplify the
    object we saved, otherwise, for example, FunctionApplication(+, [2,2]) would not be a FunctionApplication,
    but a Constant of 4.

    There are two ways we can do this in SymPy:
    1. set evaluate=False in the arguments when creating an expression
    2. Use UnevaluatedExpr() class in sympy.

    However, each has its own limitation, which will be shown in this test function.
    """

    def make_evaluated(func, args):
        return func(*args)

    def make_unevaluated1(func, args):  # approach 1
        return func(*args, evaluate=False)

    def make_unevaluated2(func, args):  # approach 2
        return func(*[sympy.UnevaluatedExpr(arg) for arg in args])

    # Approach #1
    x, y = symbols("x y")
    add0 = make_evaluated(sympy.Add, [x, x])
    assert add0 == 2 * x
    assert not add0.func.is_Add
    assert add0.func.is_Mul

    # if evaluate=False, internal representation is "x+x" instead of "2*x"
    add1 = make_unevaluated1(sympy.Add, [x, x])
    assert add1.func.is_Add
    assert not add1.func.is_Mul

    # limitation: cannot be used for Lambda.
    # Lambda is useful for representing some simple operations that sympy does not support natively such as minus
    minus = sympy.Lambda((x, y), x - y)
    assert make_evaluated(minus, [3, 1]) == 2
    assert make_evaluated(minus, [x, 1]) == x - 1
    with pytest.raises(TypeError):
        make_unevaluated1(
            minus, [3, 1]
        )  # minus(3, 1, evaluate=False) <- no evaluate argument
    with pytest.raises(TypeError):
        sympy.Lambda((x, y), x - y, evaluate=False)  # this also does not work

    # Approach #2
    minus1 = make_unevaluated2(minus, [4, 1])
    minus2 = make_unevaluated2(minus, [x, x])
    assert len(minus1.args) == 2
    assert minus1 != 3
    assert minus1.simplify() == 3  # 4-1 == 3
    # Limitation#1: cannot force unevaluated outside the UnevaluatedExpr()
    # so UnevaluatedExpr(x) - UnevaluatedExpr(x) would still be 0
    assert minus2.args == ()
    # limitation#2: cannot force unevaluated on numbers in Min/Max
    with pytest.raises(ValueError):
        # Min/Max in sympy requires all number argument to be comparable, by making an argument UnevaluatedExpr,
        # we are making it incomparable.
        make_unevaluated2(
            sympy.Min, [3, 1]
        )  # error message: "3" must be comparable
    # However approach #1 works
    assert len(make_unevaluated1(sympy.Min, [3, 1]).args) == 2
    assert make_unevaluated1(sympy.Min, [3, 1]) != 1
    # multiples ways to evaluate
    assert make_unevaluated1(sympy.Min, [3, 1]).doit() == 1
    assert make_unevaluated1(sympy.Min, [3, 1]).simplify() == 1
    assert make_evaluated(sympy.Min, [3, 1]) == 1

    # I tried this but it does not work
    minus1 = sympy.Lambda((x, y), sympy.Add(x, -y, evaluate=False))
    assert minus1(x, x) == 0
    # it still evaluates: since in Lambda's __call__ they called xreplace() see test_replace() below.


def test_replace():
    # replace in sympy is annoying because it always try to simplify the expression.
    x, y = symbols("x y")
    add0 = sympy.Add(x, -x, evaluate=False)
    assert len(add0.args) == 2
    assert add0 != 0

    assert add0.replace(x, y) == 0
    assert add0.replace(x, sympy.UnevaluatedExpr(y)) == 0
    assert add0.replace(x, sympy.UnevaluatedExpr(x)) == 0
    assert (
        add0.replace(x, x) != 0
    )  # only this is not 0 because it does nothing

    assert add0.subs(x, y) == 0
    assert add0.xreplace({x: y}) == 0
    # so the only clean solution I can think of is to set global_parameters of sympy, which is thread-safe
    # (see sympy.core.parameter)
    # from sympy.core.parameters import global_parameters as gp
