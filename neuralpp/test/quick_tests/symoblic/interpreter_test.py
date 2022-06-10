import pytest
import operator

from neuralpp.symbolic.basic_expression import BasicVariable, BasicConstant, BasicFunctionApplication
from neuralpp.symbolic.basic_interpreter import BasicInterpreter


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
