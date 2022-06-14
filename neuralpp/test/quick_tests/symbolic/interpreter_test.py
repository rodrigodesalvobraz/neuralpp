import pytest
import operator
import sympy
import builtins

from neuralpp.symbolic.basic_expression import BasicVariable, BasicConstant, BasicFunctionApplication
from neuralpp.symbolic.basic_interpreter import BasicInterpreter
from neuralpp.symbolic.sympy_expression import SymPyVariable, SymPyConstant, SymPyFunctionApplication, \
    SymPyExpression, is_sympy_value
from neuralpp.symbolic.sympy_interpreter import SymPyInterpreter


def test_basic_interpreter():
    true_context = BasicConstant(True)
    bi = BasicInterpreter()

    # cannot evaluate variable
    a = BasicVariable("a")
    with pytest.raises(AttributeError):
        bi.eval(a, true_context)

    # trivial case to evaluate constant
    one = BasicConstant(1)
    assert bi.eval(one, true_context) == 1

    # lambda as a function
    lambda_add = BasicConstant(lambda x, y: x + y)
    add_one_to_one = BasicFunctionApplication(lambda_add, [one, one])
    assert bi.eval(add_one_to_one, true_context) == 2

    python_divide = BasicConstant(lambda x, y: x / y)
    divide_by_zero = BasicFunctionApplication(python_divide, [one, BasicConstant(0)])
    with pytest.raises(ZeroDivisionError):
        bi.eval(divide_by_zero, true_context)

    # operator
    operator_add = BasicConstant(operator.add)
    built_in_add_one_to_two = BasicFunctionApplication(operator_add, [one, BasicConstant(2)])
    assert bi.eval(built_in_add_one_to_two, true_context) == 3

    # uninterpreted function
    uninterpreted_func = BasicVariable("func")
    uninterpreted_application = BasicFunctionApplication(uninterpreted_func, [one])
    with pytest.raises(AttributeError):
        bi.eval(uninterpreted_application, true_context)

    # a nested function application example
    nested_add = BasicFunctionApplication(lambda_add, [one, add_one_to_one])
    assert bi.eval(nested_add, true_context) == 3


def dict_to_sympy_context(kv_map: dict) -> SymPyExpression:
    result = sympy.S.true
    for k, v in kv_map.items():
        result = sympy.And(result, sympy.Eq(sympy.symbols(k), v, evaluate=False))
    return SymPyFunctionApplication(result)


def test_sympy_interpreter():
    true_context = SymPyConstant(sympy.S.true)
    si = SymPyInterpreter()

    # trivial case to evaluate constant
    one = SymPyConstant(sympy.Integer(1))
    assert si.eval(one, true_context) == 1

    # operator
    operator_add = BasicConstant(operator.add)
    two = SymPyConstant(sympy.Integer(2))
    add_one_to_two = SymPyExpression.new_function_application(operator_add, [one, two])
    assert si.eval(add_one_to_two, true_context) == 3

    # cannot evaluate variable
    a = SymPyVariable(sympy.symbols("a"))
    with pytest.raises(RuntimeError):
        si.eval(a, true_context)

    # more interesting cases where there is a context
    x, y, z = sympy.symbols("x y z")
    assert si.eval(SymPyFunctionApplication(x * 3), SymPyFunctionApplication(sympy.Eq(x, 10, evaluate=False))) == 30

    dict1 = {"x": 3, "y": 5}
    assert si.eval(SymPyFunctionApplication(x * y), dict_to_sympy_context(dict1)) == 15

    dict2 = {"x": 3, "y": 5, "z": 100}
    assert si.eval(SymPyFunctionApplication(x * y + z), dict_to_sympy_context(dict2)) == 115

    # test operators
    operator_mul = BasicConstant(operator.mul)
    two_times_two = SymPyExpression.new_function_application(operator_mul, [two, two])
    assert si.eval(two_times_two, true_context) == 4

    operator_pow = BasicConstant(operator.pow)
    three = SymPyConstant(sympy.Integer(3))
    two_to_the_third = SymPyExpression.new_function_application(operator_pow, [two, three])
    assert si.eval(two_to_the_third, true_context) == 8

    operator_and = BasicConstant(operator.__and__)
    true = SymPyConstant(sympy.S.true)
    false = SymPyConstant(sympy.S.false)
    true_and_false = SymPyExpression.new_function_application(operator_and, [true, false])
    assert not si.eval(true_and_false, true_context)

    true_or_false = SymPyExpression.new_function_application(BasicConstant(operator.__or__), [true, false])
    assert si.eval(true_or_false, true_context)

    not_true = SymPyExpression.new_function_application(BasicConstant(operator.__not__), [true])
    assert not si.eval(not_true, true_context)

    one_le_one = SymPyExpression.new_function_application(BasicConstant(operator.le), [one, one])
    assert si.eval(one_le_one, true_context)

    one_lt_one = SymPyExpression.new_function_application(BasicConstant(operator.lt), [one, one])
    assert not si.eval(one_lt_one, true_context)

    max_of_one_three = SymPyExpression.new_function_application(BasicConstant(builtins.max), [one, three])
    assert si.eval(max_of_one_three, true_context) == 3

    min_of_three_five = SymPyExpression.new_function_application(BasicConstant(builtins.min),
                                                                 [SymPyVariable(x), SymPyVariable(y)])
    assert si.eval(min_of_three_five, dict_to_sympy_context(dict1)) == 3
