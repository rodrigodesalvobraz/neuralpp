import fractions

import pytest
import operator
import sympy
import builtins
import z3
import math

from typing import Callable
from neuralpp.symbolic.basic_expression import BasicFunctionApplication, BasicConstant, BasicVariable, \
    BasicExpression
from neuralpp.symbolic.sympy_expression import SymPyConstant, SymPyExpression, python_callable_to_sympy_function, \
    sympy_function_to_python_callable, SymPyFunctionApplication
from neuralpp.symbolic.z3_expression import Z3FunctionApplication, Z3Constant, Z3Variable, Z3Expression


@pytest.fixture(params=[BasicExpression, SymPyExpression, Z3Expression])
def expression_factory(request):
    return request.param


int_to_int_to_int = Callable[[int, int], int]  # int -> int -> int


def test_constant(expression_factory):
    # Constant can be anything
    constant_one = expression_factory.new_constant(1)
    constant_abc = expression_factory.new_constant("abc", Callable[[int], int])
    assert constant_one != constant_abc
    assert constant_one == expression_factory.new_constant(2-1)
    assert constant_one.subexpressions == []
    assert not constant_one.contains(constant_abc)
    assert constant_one.contains(constant_one)  # A constant contains() itself
    assert constant_one.replace(constant_one, constant_abc) == constant_abc
    with pytest.raises(IndexError):
        constant_one.set(1, constant_abc)
    assert constant_one.value == 1
    one_third = fractions.Fraction(1, 3)
    constant_real = expression_factory.new_constant(one_third)
    assert constant_real.value == one_third
    constant_true = expression_factory.new_constant(True)
    constant_false = expression_factory.new_constant(False)
    assert constant_true.value
    assert not constant_false.value


def test_basic_constant():
    constant_abc = BasicConstant("abc")
    assert constant_abc.value == "abc"


def test_sympy_constant():
    constant_sympy_obj = SymPyConstant(sympy.Rational(1, 3), float)
    constant_sympy_obj2 = SymPyConstant(sympy.Rational(3, 9), float)
    assert constant_sympy_obj == constant_sympy_obj2
    constant_abc = SymPyExpression.new_constant("abc", int_to_int_to_int)
    assert constant_abc.value == "abc"


def test_variable(expression_factory):
    # Variable is initialized by a variable name.
    variable_x = expression_factory.new_variable("x", int)
    variable_y = expression_factory.new_variable("y", int)
    assert variable_x != variable_y
    assert variable_x == expression_factory.new_variable("x", int)
    assert variable_x != expression_factory.new_variable("x", bool)  # must be of the same type
    assert variable_x.subexpressions == []
    assert not variable_x.contains(variable_y)
    assert variable_x.contains(variable_x)
    assert variable_x.replace(variable_x, variable_y) == variable_y
    with pytest.raises(IndexError):
        variable_x.set(1, variable_y)
    assert variable_x.name == "x"
    assert variable_y.name == "y"


def test_basic_function_application():
    int_type = int

    # function application has the interface FunctionApplication(func: Expression, args: List[Expression]).
    # Note that the first argument `func` is of Expression type.
    # There are 2 possible choices of `func` for a meaningful BasicFunctionApplication:
    # 1. a Python Callable
    func1 = BasicConstant(lambda x, y: x + y, int_to_int_to_int)
    constant_one = BasicConstant(1)
    constant_two = BasicConstant(2)
    fa1 = BasicFunctionApplication(func1, [constant_one, constant_two])
    # type of function application (i.e., the return type) is optional
    assert fa1 == BasicFunctionApplication(func1, [constant_one, constant_two])
    assert BasicFunctionApplication(func1, [constant_one, constant_two]).type == int_type
    # using operator.add is easy to compare (operator.add == operator.add) and thus more desirable than using lambda.
    func2 = BasicConstant(operator.add, int_to_int_to_int)
    fa2 = BasicFunctionApplication(func2, [constant_one, constant_two])

    # 2. a Variable. In this case the function is uninterpreted. Even it's named "add", it does not necessarily
    # have to be a function of addition.
    func3 = BasicVariable("add", int_to_int_to_int)
    # use fa2 here, expression can be recursive
    fa3 = BasicFunctionApplication(func3, [constant_one, fa2])

    assert fa2.subexpressions == [BasicConstant(operator.add, int_to_int_to_int), constant_one, constant_two]
    assert fa2 == fa2
    # python cannot check __eq__ of two lambdas, so we have the following inequality
    assert fa1.subexpressions != [lambda x, y: x + y, constant_one, constant_two]
    # but if the two lambdas are the same object then they are equal.
    assert fa1 == fa1

    assert fa1 != fa2
    assert fa2 != fa3
    assert fa2 != BasicFunctionApplication(func2, [constant_one, constant_one])
    assert fa2 != BasicFunctionApplication(func2, [constant_one, fa2])
    assert fa1.contains(constant_two)
    assert fa3.contains(constant_two)  # shows that search of contains() is deep.

    # replace() is also deep and not in-place (returns a new object instead of modify the called on)
    fa4 = fa3.replace(constant_two, constant_one)
    assert fa4 != fa3
    assert fa3 == BasicFunctionApplication(func3, [constant_one, fa2])
    assert fa4 == BasicFunctionApplication(func3, [constant_one,
                                                   BasicFunctionApplication(func2, [constant_one, constant_one])])

    # set() is also not changing the object that it is called on
    fa5 = fa2.set(0, BasicVariable("f", int_to_int_to_int))
    assert fa5 == BasicFunctionApplication(BasicVariable("f", int_to_int_to_int), [constant_one, constant_two])
    assert fa2 == BasicFunctionApplication(func2, [constant_one, constant_two])
    fa6 = fa2.set(1, BasicVariable("a", int_type))
    assert fa6 == BasicFunctionApplication(func2, [BasicVariable("a", int_type), constant_two])
    with pytest.raises(IndexError):
        fa4.set(3, constant_one)

    fa7 = fa4.replace(constant_one, constant_two)
    assert fa7 == BasicFunctionApplication(func3, [constant_two,
                                                   BasicFunctionApplication(func2, [constant_two, constant_two])])


@pytest.fixture(params=[operator.and_, operator.or_, operator.invert, operator.xor, operator.le,
                        operator.lt, operator.ge, operator.gt, operator.eq,
                        operator.add, operator.mul, operator.pow, builtins.min, builtins.max])
def python_callable(request):
    return request.param


@pytest.fixture(params=[sympy.And, sympy.Or, sympy.Not, sympy.Xor, sympy.Le, sympy.Lt, sympy.Ge, sympy.Gt, sympy.Eq,
                        sympy.Add, sympy.Mul, sympy.Pow, sympy.Min, sympy.Max])
def sympy_func(request):
    return request.param


def test_python_callable_and_sympy_function_conversion(sympy_func):
    assert python_callable_to_sympy_function(sympy_function_to_python_callable(sympy_func)) == sympy_func


def test_python_callable_and_sympy_function_conversion2(python_callable):
    assert sympy_function_to_python_callable(python_callable_to_sympy_function(python_callable)) == python_callable


def test_function_application(expression_factory):
    """ General test cases for function application. """
    constant_one = SymPyExpression.new_constant(1)
    constant_two = SymPyExpression.new_constant(2)
    add_func = SymPyExpression.new_constant(operator.add, int_to_int_to_int)
    fa = SymPyExpression.new_function_application(add_func, [constant_one, constant_two])

    assert fa.subexpressions[0] == add_func
    assert type(constant_one) == SymPyConstant
    assert type(fa.subexpressions[1]) == SymPyConstant
    assert fa.subexpressions[1].value == 1
    assert fa.subexpressions[2].value == 2
    assert fa == fa

    assert fa != SymPyExpression.new_function_application(add_func, [constant_one, constant_one])
    assert fa == SymPyExpression.new_function_application(add_func, [constant_one, constant_two])

    assert constant_one == SymPyConstant(sympy.Integer(1))

    fa2 = SymPyExpression.new_function_application(add_func, [constant_one, fa])
    fa2 = fa2.replace(constant_one, constant_two)
    assert fa2 == SymPyExpression.new_function_application(
        add_func, [constant_two, SymPyExpression.new_function_application(add_func, [constant_two, constant_two])])

    # some new type test.
    assert fa2.subexpressions[0].type == int_to_int_to_int
    assert fa2.subexpressions[1].type == int
    assert fa2.subexpressions[2].type == int  # return type
    assert fa2.subexpressions[2].function.type == int_to_int_to_int  # return type
    assert fa2.subexpressions[2].subexpressions[1].type == int
    assert fa2.subexpressions[2].subexpressions[2].type == int

    fa3 = fa.set(0, SymPyExpression.new_constant(operator.mul, int_to_int_to_int))
    assert fa3 == SymPyExpression.new_function_application(SymPyExpression.new_constant(operator.mul,
                                                                                        int_to_int_to_int),
                                                           [constant_one, constant_two])
    assert fa == SymPyExpression.new_function_application(add_func, [constant_one, constant_two])
    fa6 = fa.set(1, SymPyExpression.new_variable("a", int))
    assert fa6 == SymPyExpression.new_function_application(add_func, [SymPyExpression.new_variable("a", int),
                                                                      constant_two])
    with pytest.raises(IndexError):
        fa2.set(3, constant_one)

    # operator overloading
    x = BasicVariable("x", int)
    x_plus_one = x + 1
    assert x_plus_one.function.type == int_to_int_to_int
    one_third_plus_x = fractions.Fraction(1, 3) + x
    assert one_third_plus_x.function.type == Callable[[fractions.Fraction, int], fractions.Fraction]
    x_minus_one = x - 1
    assert x_minus_one.function.type == int_to_int_to_int
    pi_times_x = math.pi * x
    assert pi_times_x.function.type == Callable[[float, int], float]


def test_sympy_function_application():
    constant_two = SymPyExpression.new_constant(2)
    # a trickier one: add can be of different types.
    int_to_float_to_float = Callable[[int, float], float]
    float_to_float_to_float = Callable[[float, float], float]
    # this one does not work on z3 because z3 enforce arguments of add to have the same type.
    mixed_add_func = SymPyExpression.new_constant(operator.add, int_to_float_to_float)
    float_add_func = SymPyExpression.new_constant(operator.add, float_to_float_to_float)
    constant_two_f = SymPyExpression.new_constant(2.0)
    mixed_adds = SymPyExpression.new_function_application(
        mixed_add_func, [constant_two,
                         SymPyExpression.new_function_application(float_add_func,
                                                                  [constant_two_f, constant_two_f])])
    assert mixed_adds.type == float
    assert mixed_adds.subexpressions[0].type == int_to_float_to_float
    assert mixed_adds.subexpressions[1].type == int
    assert mixed_adds.subexpressions[2].type == float
    assert mixed_adds.subexpressions[2].function.type == float_to_float_to_float
    assert mixed_adds.subexpressions[2].subexpressions[1].type == float
    assert mixed_adds.subexpressions[2].subexpressions[2].type == float


def test_z3_function_application():
    """
    We cannot mimic the test above because we do not support z3's float in function yet.
    (see z3_usage_test.py: test_z3_fp_sort())
    Also, z3 required add/compare/... to be of same type, so we cannot have "add: real -> int -> real".
    In that case explicit conversion is requires (z3.ToInt()/z3.ToReal())
    """
    real = fractions.Fraction
    real_to_real_to_real = Callable[[real, real], real]
    real_add_func = Z3Expression.new_constant(operator.add, real_to_real_to_real)
    z3_constant_one_third_r = z3.RealVal(fractions.Fraction(1, 3))
    constant_one_third_r = Z3Constant(z3_constant_one_third_r)

    add0 = Z3FunctionApplication(z3_constant_one_third_r + z3_constant_one_third_r)
    add1 = Z3Expression.new_function_application(real_add_func, [constant_one_third_r, add0])
    assert add1.type == real
    assert add1.subexpressions[0].type == real_to_real_to_real
    assert add1.subexpressions[1].type == real
    assert add1.subexpressions[2].type == real
    assert add1.subexpressions[2].function.type == real_to_real_to_real
    assert add1.subexpressions[2].subexpressions[1].type == real
    assert add1.subexpressions[2].subexpressions[2].type == real


def test_sympy_z3_conversion():
    real = fractions.Fraction
    real_to_real_to_real = Callable[[real, real], real]

    real_add_func = Z3Expression.new_constant(operator.add, real_to_real_to_real)
    z3_constant_one_third_r = z3.RealVal(fractions.Fraction(1, 3))
    constant_one_third_r = Z3Constant(z3_constant_one_third_r)

    add0 = Z3FunctionApplication(z3_constant_one_third_r + z3_constant_one_third_r)
    add1 = Z3Expression.new_function_application(real_add_func, [constant_one_third_r, add0])

    sympy_real_add_func = SymPyExpression.new_constant(operator.add, real_to_real_to_real)
    sympy_constant_one_third_r = sympy.Rational(fractions.Fraction(1, 3))
    constant_one_third2 = SymPyConstant(sympy_constant_one_third_r)

    sympy_add = sympy.Add(sympy_constant_one_third_r, sympy_constant_one_third_r, evaluate=False)
    two_third = SymPyFunctionApplication(sympy_add, {}, real_to_real_to_real)

    # in creating add2, we implicitly convert the sympy child `two_third`.
    add2 = Z3Expression.new_function_application(real_add_func, [constant_one_third2, two_third])
    assert add2.arguments[1] == add0
    assert add2 == add1
