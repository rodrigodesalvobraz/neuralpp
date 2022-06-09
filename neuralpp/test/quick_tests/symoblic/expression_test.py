import pytest

from neuralpp.symbolic.function import Add
from neuralpp.symbolic.basic_expression import BasicFunctionApplication, BasicConstant, BasicVariable
    

def test_basic_constant():
    # Constant can be anything
    constant_one = BasicConstant(1)
    constant_abc = BasicConstant("abc")
    assert constant_one != constant_abc
    assert constant_one == BasicConstant(2-1)
    assert constant_one.subexpressions() == []
    assert not constant_one.contains(constant_abc)
    assert constant_one.contains(constant_one)  # A constant contains() itself
    assert constant_one.replace(constant_one, constant_abc) == constant_abc
    with pytest.raises(IndexError):
        constant_one.set(1, constant_abc)
    assert constant_one.value == 1
    assert constant_abc.value == "abc"


def test_basic_variable():
    # Variable is initialized by a variable name.
    variable_x = BasicVariable("x")
    variable_y = BasicVariable("y")
    assert variable_x != variable_y
    assert variable_x == BasicVariable("x")
    assert variable_x.subexpressions() == []
    assert not variable_x.contains(variable_y)
    assert variable_x.contains(variable_x)  # A variable contains() itself
    assert variable_x.replace(variable_x, variable_y) == variable_y
    with pytest.raises(IndexError):
        variable_x.set(1, variable_y)
    assert variable_x.name == "x"
    assert variable_y.name == "y"


def test_basic_function_application():
    # function application has the interface FunctionApplication(func: Expression, args: List[Expression]).
    # Note that the first argument `func` is of Expression type.
    # There are 3 possible choices of `func` (for a meaningful FunctionApplication):

    # 1. a Python Callable
    func1 = BasicConstant(lambda x, y: x + y)
    constant_one = BasicConstant(1)
    constant_two = BasicConstant(2)
    fa1 = BasicFunctionApplication(func1, [constant_one, constant_two])

    # 2. a Function. Here Add is a subclass of Function. All Function has a method lambdify(), which
    # will be handy for interpretation.
    func2 = BasicConstant(Add())
    fa2 = BasicFunctionApplication(func2, [constant_one, constant_two])

    # 3. a Variable. In this case the function is uninterpreted. Even it's named "add", it does not necessarily
    # have to be a function of addition.
    func3 = BasicConstant(BasicVariable("add"))
    fa3 = BasicFunctionApplication(func3, [constant_one, fa2])  # use fa1 here, expression can be recursive

    assert fa2.subexpressions() == [func2, constant_one, constant_two]
    assert fa2 == fa2
    # python cannot check __eq__ of two lambdas, so we have the following inequality
    assert fa1.subexpressions() != [lambda x, y: x + y, constant_one, constant_two]
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
    fa5 = fa2.set(0, BasicVariable("f"))
    assert fa5 == BasicFunctionApplication(BasicVariable("f"), [constant_one, constant_two])
    assert fa2 == BasicFunctionApplication(func2, [constant_one, constant_two])
    fa6 = fa2.set(1, BasicVariable("a"))
    assert fa6 == BasicFunctionApplication(func2, [BasicVariable("a"), constant_two])
    with pytest.raises(IndexError):
        fa4.set(3, constant_one)
