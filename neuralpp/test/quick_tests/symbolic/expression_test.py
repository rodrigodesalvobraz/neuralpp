import pytest
import operator
import sympy

from neuralpp.symbolic.basic_expression import BasicFunctionApplication, BasicConstant, BasicVariable, \
    BasicExpression
from neuralpp.symbolic.sympy_expression import SymPyFunctionApplication, SymPyConstant, SymPyVariable, \
    SymPyExpression


@pytest.fixture(params=[BasicExpression, SymPyExpression])
def expression_factory(request):
    return request.param


def test_constant(expression_factory):
    # Constant can be anything
    constant_one = expression_factory.new_constant(1)
    constant_abc = expression_factory.new_constant("abc")
    assert constant_one != constant_abc
    assert constant_one == expression_factory.new_constant(2-1)
    assert constant_one.subexpressions == []
    assert not constant_one.contains(constant_abc)
    assert constant_one.contains(constant_one)  # A constant contains() itself
    assert constant_one.replace(constant_one, constant_abc) == constant_abc
    with pytest.raises(IndexError):
        constant_one.set(1, constant_abc)
    assert constant_one.value == 1
    constant_float = expression_factory.new_constant(3.14)
    assert constant_float.value == 3.14
    constant_true = expression_factory.new_constant(True)
    constant_false = expression_factory.new_constant(False)
    assert constant_true.value
    assert not constant_false.value


def test_basic_constant():
    constant_abc = BasicConstant("abc")
    assert constant_abc.value == "abc"


def test_sympy_constant():
    constant_sympy_obj = SymPyConstant(sympy.Rational(1, 3))
    constant_sympy_obj2 = SymPyConstant(sympy.Rational(3, 9))
    assert constant_sympy_obj == constant_sympy_obj2
    constant_abc = SymPyExpression.new_constant("abc")
    assert constant_abc.value == sympy.Function("abc")


def test_variable(expression_factory):
    # Variable is initialized by a variable name.
    variable_x = expression_factory.new_variable("x")
    variable_y = expression_factory.new_variable("y")
    assert variable_x != variable_y
    assert variable_x == expression_factory.new_variable("x")
    assert variable_x.subexpressions == []
    assert not variable_x.contains(variable_y)
    assert variable_x.contains(variable_x)
    assert variable_x.replace(variable_x, variable_y) == variable_y
    with pytest.raises(IndexError):
        variable_x.set(1, variable_y)
    assert variable_x.name == "x"
    assert variable_y.name == "y"


def test_basic_function_application():
    # function application has the interface FunctionApplication(func: Expression, args: List[Expression]).
    # Note that the first argument `func` is of Expression type.
    # There are 2 possible choices of `func` for a meaningful BasicFunctionApplication:

    # 1. a Python Callable
    func1 = BasicConstant(lambda x, y: x + y)
    constant_one = BasicConstant(1)
    constant_two = BasicConstant(2)
    fa1 = BasicFunctionApplication(func1, [constant_one, constant_two])
    # using operator.add is easy to compare (operator.add == operator.add) and thus more desirable than using lambda.
    func2 = BasicConstant(operator.add)
    fa2 = BasicFunctionApplication(func2, [constant_one, constant_two])

    # 2. a Variable. In this case the function is uninterpreted. Even it's named "add", it does not necessarily
    # have to be a function of addition.
    func3 = BasicConstant(BasicVariable("add"))
    fa3 = BasicFunctionApplication(func3, [constant_one, fa2])  # use fa2 here, expression can be recursive

    assert fa2.subexpressions == [BasicConstant(operator.add), constant_one, constant_two]
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
    fa5 = fa2.set(0, BasicVariable("f"))
    assert fa5 == BasicFunctionApplication(BasicVariable("f"), [constant_one, constant_two])
    assert fa2 == BasicFunctionApplication(func2, [constant_one, constant_two])
    fa6 = fa2.set(1, BasicVariable("a"))
    assert fa6 == BasicFunctionApplication(func2, [BasicVariable("a"), constant_two])
    with pytest.raises(IndexError):
        fa4.set(3, constant_one)

    fa7 = fa4.replace(constant_one, constant_two)
    assert fa7 == BasicFunctionApplication(func3, [constant_two,
                                                   BasicFunctionApplication(func2, [constant_two, constant_two])])


def test_sympy_function_application():
    constant_one = SymPyExpression.new_constant(1)
    constant_two = SymPyExpression.new_constant(2)
    add_func = SymPyExpression.new_constant(operator.add)
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

    fa3 = fa.set(0, SymPyExpression.new_constant(operator.mul))
    assert fa3 == SymPyExpression.new_function_application(SymPyExpression.new_constant(operator.mul),
                                                           [constant_one, constant_two])
    assert fa == SymPyExpression.new_function_application(add_func, [constant_one, constant_two])
    fa6 = fa.set(1, SymPyExpression.new_variable("a"))
    assert fa6 == SymPyExpression.new_function_application(add_func, [SymPyExpression.new_variable("a"), constant_two])
    with pytest.raises(IndexError):
        fa2.set(3, constant_one)
