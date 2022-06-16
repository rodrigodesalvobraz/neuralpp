import pytest
import operator
import sympy
import builtins

from neuralpp.symbolic.expression import FunctionType
from neuralpp.symbolic.basic_expression import BasicVariable, BasicConstant, BasicFunctionApplication, new_type
from neuralpp.symbolic.basic_interpreter import BasicInterpreter
from neuralpp.symbolic.sympy_expression import SymPyVariable, SymPyConstant, SymPyFunctionApplication, \
    SymPyExpression
from neuralpp.symbolic.sympy_interpreter import SymPyInterpreter


int_to_int_to_int = BasicConstant(FunctionType([int, int], int))
int_to_int_to_bool = BasicConstant(FunctionType([int, int], bool))
bool_to_bool_to_bool = BasicConstant(FunctionType([bool, bool], bool))


def test_basic_interpreter():
    true_context = BasicConstant(True)
    bi = BasicInterpreter()

    # cannot evaluate variable
    a = BasicVariable("a", new_type(int))
    with pytest.raises(AttributeError):
        bi.eval(a, true_context)

    # trivial case to evaluate constant
    one = BasicConstant(1)
    assert bi.eval(one, true_context) == 1

    # lambda as a function
    lambda_add = BasicConstant(lambda x, y: x + y, int_to_int_to_int)
    add_one_to_one = BasicFunctionApplication(lambda_add, [one, one])
    assert bi.eval(add_one_to_one, true_context) == 2

    python_divide = BasicConstant(lambda x, y: x / y, int_to_int_to_int)
    divide_by_zero = BasicFunctionApplication(python_divide, [one, BasicConstant(0)])
    with pytest.raises(ZeroDivisionError):
        bi.eval(divide_by_zero, true_context)

    # operator
    operator_add = BasicConstant(operator.add, int_to_int_to_int)
    built_in_add_one_to_two = BasicFunctionApplication(operator_add, [one, BasicConstant(2)])
    assert bi.eval(built_in_add_one_to_two, true_context) == 3

    # uninterpreted function
    uninterpreted_func = BasicVariable("func", int_to_int_to_int)
    uninterpreted_application = BasicFunctionApplication(uninterpreted_func, [one])
    with pytest.raises(AttributeError):
        bi.eval(uninterpreted_application, true_context)

    # a nested function application example
    nested_add = BasicFunctionApplication(lambda_add, [one, add_one_to_one])
    assert bi.eval(nested_add, true_context) == 3


def boolean_function_of_arity(arity: int) -> BasicConstant:
    return BasicConstant(FunctionType([bool for i in range(arity)], bool))


def dict_to_sympy_context(kv_map: dict) -> SymPyExpression:
    result = sympy.S.true
    type_dict = {}
    for k, v in kv_map.items():
        symbol = sympy.symbols(k)
        eq_expression = sympy.Eq(symbol, v, evaluate=False)
        result = sympy.And(result, eq_expression)
        type_dict[symbol] = new_type(int)  # it's fine for test, we only use int
        type_dict[eq_expression] = int_to_int_to_bool
    return SymPyFunctionApplication(result, type_dict, boolean_function_of_arity(len(result.args)))


def test_sympy_interpreter():
    true_context = SymPyConstant(sympy.S.true)
    si = SymPyInterpreter()

    # trivial case to evaluate constant
    one = SymPyConstant(sympy.Integer(1))
    assert si.eval(one, true_context) == 1

    # operator
    operator_add = BasicConstant(operator.add, int_to_int_to_int)
    two = SymPyConstant(sympy.Integer(2))
    add_one_to_two = SymPyExpression.new_function_application(operator_add, [one, two])
    assert si.eval(add_one_to_two, true_context) == 3

    # cannot evaluate variable
    a = SymPyVariable(sympy.symbols("a"), new_type(int))
    with pytest.raises(RuntimeError):
        si.eval(a, true_context)

    # more interesting cases where there is a context
    x, y, z = sympy.symbols("x y z")
    assert si.eval(SymPyFunctionApplication(x * 3, {x: new_type(int)}, int_to_int_to_int),
                   SymPyFunctionApplication(sympy.Eq(x, 10, evaluate=False), {x: new_type(int)}, int_to_int_to_bool)) \
           == 30

    dict1 = {"x": 3, "y": 5}
    assert si.eval(SymPyFunctionApplication(x * y, {x: new_type(int), y: new_type(int)}, int_to_int_to_int),
                   dict_to_sympy_context(dict1)) == 15

    dict2 = {"x": 3, "y": 5, "z": 100}
    assert si.eval(SymPyFunctionApplication(x * y + z, {x: new_type(int), y: new_type(int), z: new_type(int),
                                                        x*y: int_to_int_to_int},
                                            int_to_int_to_int),
                   dict_to_sympy_context(dict2)) == 115

    # test operators
    operator_mul = BasicConstant(operator.mul, int_to_int_to_int)
    two_times_two = SymPyExpression.new_function_application(operator_mul, [two, two])
    assert si.eval(two_times_two, true_context) == 4

    operator_pow = BasicConstant(operator.pow, int_to_int_to_int)
    three = SymPyConstant(sympy.Integer(3))
    two_to_the_third = SymPyExpression.new_function_application(operator_pow, [two, three])
    assert si.eval(two_to_the_third, true_context) == 8

    operator_and = BasicConstant(operator.__and__, bool_to_bool_to_bool)
    true = SymPyConstant(sympy.S.true)
    false = SymPyConstant(sympy.S.false)
    with pytest.raises(TypeError):
        # we cannot do this because there's no way to prevent SymPy to evaluate And(True,False)
        true_and_false = SymPyExpression.new_function_application(operator_and, [true, false])
    # even if we specify evaluate=False, it still evaluates.
    assert not sympy.And(sympy.S.true, sympy.S.false, evaluate=False)

    # it's the case for or
    with pytest.raises(TypeError):
        true_or_false = SymPyExpression.new_function_application(BasicConstant(operator.__or__), [true, false])
    # but not the case for "not" (for sympy 1.10.1, wonder if it's a bug?)
    not_true = SymPyExpression.new_function_application(BasicConstant(operator.__not__), [true])
    assert not si.eval(not_true, true_context)

    one_le_one = SymPyExpression.new_function_application(BasicConstant(operator.le, int_to_int_to_bool), [one, one])
    assert si.eval(one_le_one, true_context)

    one_lt_one = SymPyExpression.new_function_application(BasicConstant(operator.lt, int_to_int_to_bool), [one, one])
    assert not si.eval(one_lt_one, true_context)

    max_of_one_three = SymPyExpression.new_function_application(BasicConstant(builtins.max, int_to_int_to_int),
                                                                [one, three])
    assert si.eval(max_of_one_three, true_context) == 3

    min_of_three_five = SymPyExpression.new_function_application(BasicConstant(builtins.min, int_to_int_to_int),
                                                                 [SymPyVariable(x, new_type(int)),
                                                                  SymPyVariable(y, new_type(int))])
    assert si.eval(min_of_three_five, dict_to_sympy_context(dict1)) == 3
