import fractions

import pytest
import operator
import sympy
import builtins

from typing import Callable
from neuralpp.symbolic.basic_expression import BasicVariable, BasicConstant, BasicFunctionApplication
from neuralpp.symbolic.basic_interpreter import BasicInterpreter
from neuralpp.symbolic.sympy_expression import SymPyVariable, SymPyConstant, SymPyFunctionApplication, \
    SymPyExpression
from neuralpp.symbolic.sympy_interpreter import SymPyInterpreter


int_to_int_to_int = Callable[[int, int], int]
int_to_real_to_real = Callable[[int, fractions.Fraction], fractions.Fraction]
int_to_int_to_bool = Callable[[int, int], bool]
bool_to_bool_to_bool = Callable[[bool, bool], bool]


def test_basic_interpreter():
    bi = BasicInterpreter()

    # cannot evaluate variable
    a = BasicVariable("a", int)
    with pytest.raises(AttributeError):
        bi.eval(a)

    # trivial case to evaluate constant
    one = BasicConstant(1)
    assert bi.eval(one) == 1

    # lambda as a function
    lambda_add = BasicConstant(lambda x, y: x + y, int_to_int_to_int)
    add_one_to_one = BasicFunctionApplication(lambda_add, [one, one])
    assert bi.eval(add_one_to_one) == 2

    python_divide = BasicConstant(lambda x, y: x / y, int_to_int_to_int)
    divide_by_zero = BasicFunctionApplication(python_divide, [one, BasicConstant(0)])
    with pytest.raises(ZeroDivisionError):
        bi.eval(divide_by_zero)

    # operator
    operator_add = BasicConstant(operator.add, int_to_int_to_int)
    built_in_add_one_to_two = BasicFunctionApplication(operator_add, [one, BasicConstant(2)])
    assert bi.eval(built_in_add_one_to_two) == 3

    # uninterpreted function
    uninterpreted_func = BasicVariable("func", int_to_int_to_int)
    uninterpreted_application = BasicFunctionApplication(uninterpreted_func, [one])
    with pytest.raises(AttributeError):
        bi.eval(uninterpreted_application)

    # a nested function application example
    nested_add = BasicFunctionApplication(lambda_add, [one, add_one_to_one])
    assert bi.eval(nested_add) == 3


def boolean_function_of_arity(arity: int) -> BasicConstant:
    return Callable[[bool for i in range(arity)], bool]


def dict_to_sympy_context(kv_map: dict) -> SymPyExpression:
    conjunction = sympy.S.true
    type_dict = {}
    for k, v in kv_map.items():
        symbol = sympy.symbols(k)
        eq_expression = sympy.Eq(symbol, v, evaluate=False)
        conjunction = sympy.And(conjunction, eq_expression, evaluate=False)
        type_dict[symbol] = int  # it's fine for test, we only use int
        type_dict[eq_expression] = int_to_int_to_bool
    if len(kv_map) > 1:
        return SymPyFunctionApplication(conjunction, type_dict, boolean_function_of_arity(len(conjunction.args)))
    else:  # And(True, Eq(x,1,evaluate=False), evaluate=False) is still Eq(x,1,evaluate=False)
        return SymPyFunctionApplication(conjunction, type_dict, int_to_int_to_bool)


def test_sympy_interpreter():
    si = SymPyInterpreter()

    # trivial case to evaluate constant
    one = SymPyConstant(sympy.Integer(1))
    assert si.eval(one) == 1

    # operator
    operator_add = BasicConstant(operator.add, int_to_int_to_int)
    two = SymPyConstant(sympy.Integer(2))
    add_one_to_two = SymPyExpression.new_function_application(operator_add, [one, two])
    assert si.eval(add_one_to_two) == 3

    # cannot evaluate variable
    a = SymPyVariable(sympy.symbols("a"), int)
    with pytest.raises(RuntimeError):
        si.eval(a)

    # more interesting cases where there is a context
    x, y, z = sympy.symbols("x y z")
    assert si.eval(SymPyFunctionApplication(x * 3, {x: int}, int_to_int_to_int),
                   SymPyFunctionApplication(sympy.Eq(x, 10, evaluate=False), {x: int}, int_to_int_to_bool)) \
           == 30

    dict1 = {"x": 3, "y": 5}
    assert si.eval(SymPyFunctionApplication(x * y, {x: int, y: int}, int_to_int_to_int),
                   dict_to_sympy_context(dict1)) == 15

    dict2 = {"x": 3, "y": 5, "z": 100}
    assert si.eval(SymPyFunctionApplication(x * y + z, {x: int, y: int, z: int,
                                                        x*y: int_to_int_to_int},
                                            int_to_int_to_int),
                   dict_to_sympy_context(dict2)) == 115

    # test operators
    operator_mul = BasicConstant(operator.mul, int_to_int_to_int)
    two_times_two = SymPyExpression.new_function_application(operator_mul, [two, two])
    assert si.eval(two_times_two) == 4

    operator_pow = BasicConstant(operator.pow, int_to_int_to_int)
    three = SymPyConstant(sympy.Integer(3))
    two_to_the_third = SymPyExpression.new_function_application(operator_pow, [two, three])
    assert si.eval(two_to_the_third) == 8

    operator_and = BasicConstant(operator.and_, bool_to_bool_to_bool)
    true = SymPyConstant(sympy.S.true)
    false = SymPyConstant(sympy.S.false)
    with pytest.raises(TypeError):
        # we cannot do this because there's no way to prevent SymPy to evaluate And(True,False)
        SymPyExpression.new_function_application(operator_and, [true, false])
    # even if we specify evaluate=False, it still evaluates.
    assert not sympy.And(sympy.S.true, sympy.S.false, evaluate=False)

    # it's the case for or
    with pytest.raises(TypeError):
        SymPyExpression.new_function_application(BasicConstant(operator.or_, bool_to_bool_to_bool), [true, false])
    # but not the case for "not" (for sympy 1.10.1, wonder if it's a bug?)
    not_true = SymPyExpression.new_function_application(BasicConstant(operator.invert), [true])
    assert not si.eval(not_true)

    one_le_one = SymPyExpression.new_function_application(BasicConstant(operator.le, int_to_int_to_bool), [one, one])
    assert si.eval(one_le_one)

    one_lt_one = SymPyExpression.new_function_application(BasicConstant(operator.lt, int_to_int_to_bool), [one, one])
    assert not si.eval(one_lt_one)

    max_of_one_three = SymPyExpression.new_function_application(BasicConstant(builtins.max, int_to_int_to_int),
                                                                [one, three])
    assert si.eval(max_of_one_three) == 3

    min_of_three_five = SymPyExpression.new_function_application(BasicConstant(builtins.min, int_to_int_to_int),
                                                                 [SymPyVariable(x, int),
                                                                  SymPyVariable(y, int)])
    assert si.eval(min_of_three_five, dict_to_sympy_context(dict1)) == 3


def test_sympy_interpreter_simplify():
    si = SymPyInterpreter()
    x, y, z = sympy.symbols("x y z")
    x_plus_y = SymPyFunctionApplication(x+y, {x: int, y: int}, Callable[[int, int], int])
    neg_y = SymPyFunctionApplication(-y, {y: int}, Callable[[int, int], int])  # -y is (-1)*y in sympy
    x_plus_y_minus_y = SymPyExpression.new_function_application(BasicConstant(operator.add, int_to_int_to_int),
                                                                [x_plus_y, neg_y])
    assert x_plus_y_minus_y.number_of_arguments == 2
    assert x_plus_y_minus_y.type_dict[x] == int
    assert x_plus_y_minus_y.type_dict[y] == int
    assert x_plus_y_minus_y.type_dict[-y] == Callable[[int, int], int]
    assert x_plus_y_minus_y.type_dict[x+y] == Callable[[int, int], int]
    assert x_plus_y_minus_y.type_dict[sympy.Add(x+y, -y, evaluate=False)] == Callable[[int, int], int]
    assert len(x_plus_y_minus_y.type_dict) == 5

    x_only = si.simplify(x_plus_y_minus_y)  # simplifies x + y - y to x
    assert isinstance(x_only, SymPyVariable)
    assert x_only.type_dict == SymPyVariable(x, int).type_dict
    assert x_only == SymPyVariable(x, int)
    assert len(x_only.type_dict) == 1  # no other type information
    assert x_only.type_dict == {x: int}

    real_y = SymPyVariable(y, fractions.Fraction)
    x_plus_real_y = SymPyExpression.new_function_application(BasicConstant(operator.add, int_to_real_to_real),
                                                             [x_only, real_y])
    # old y type has been deleted
    assert len(x_plus_real_y.type_dict) == 3
    assert x_plus_real_y.type_dict[x] == int
    assert x_plus_real_y.type_dict[y] == fractions.Fraction
    assert x_plus_real_y.type_dict[sympy.Add(x, y, evaluate=False)] == \
           Callable[[int, fractions.Fraction], fractions.Fraction]

    # similar as above but with BasicExpression, internal conversion here
    b_x = BasicVariable("x", int)
    b_y = BasicVariable("y", int)
    b_x_plus_y = BasicFunctionApplication(BasicConstant(operator.add, int_to_int_to_int), [b_x, b_y])
    b_x_plus_y_minus_y = BasicFunctionApplication(BasicConstant(operator.sub, int_to_int_to_int), [b_x_plus_y, b_y])
    assert si.simplify(b_x_plus_y_minus_y) == SymPyVariable(x, int)
    assert si.simplify(b_x_plus_y).sympy_object == x + y

    # with a context
    y_2_context = dict_to_sympy_context({"y": 2})
    print(f"y=2 context:{y_2_context}")
    assert si.simplify(b_x_plus_y, y_2_context).sympy_object == x + 2
