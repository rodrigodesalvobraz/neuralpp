import pytest

import sympy
from typing import Callable
from neuralpp.inference.graphical_model.representation.factor.symbolic_factor import SymbolicFactor
from neuralpp.symbolic.sympy_expression import SymPyVariable, SymPyConstant, SymPyFunctionApplication, \
    SymPyExpression
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.symbolic.basic_expression import BasicConstant, BasicVariable, BasicFunctionApplication

int_to_int_to_int = Callable[[int, int], int]

@pytest.fixture
def x():
    return IntegerVariable("x", 3)


@pytest.fixture
def y():
    return IntegerVariable("y", 2)

@pytest.fixture
def z():
    return IntegerVariable("z", 3)

@pytest.fixture
def expression1():
    x, y = sympy.symbols("x y")
    return SymPyFunctionApplication(x * y, {x: int, y: int}, int_to_int_to_int)

@pytest.fixture
def expression2():
    x, z = sympy.symbols("x z")
    return SymPyFunctionApplication(x + z, {x: int, z: int}, int_to_int_to_int)


def test_sympy_condition(x, y, expression1):
    symbolic = SymbolicFactor([x, y], expression1)
    conditioned = symbolic.condition({x: 1})
    assert conditioned.expression == SymPyVariable(sympy.symbols("y"), int)

def test_mul_by_non_identity(x, y, z, expression1, expression2):
    symbolic1 = SymbolicFactor([x, y], expression1)
    symbolic2 = SymbolicFactor([x, z], expression2)
    symbolic3 = symbolic1.mul_by_non_identity(symbolic2)

    symbolic1_condition = symbolic1.condition({x: 1, y: 2})
    symbolic2_condition = symbolic2.condition({x: 1, z: 3})
    symbolic3_condition = symbolic3.condition({x: 1, y: 2, z: 3})

    assert symbolic1_condition.expression.sympy_object * symbolic2_condition.expression.sympy_object == symbolic3_condition.expression.sympy_object

def test_sum_out_variable(x, y, expression1):
    symbolic = SymbolicFactor([x, y], expression1)
    sum_out_x = symbolic.sum_out_variable(x)
    assert sum_out_x.expression.sympy_object == 3 * sympy.symbols("y")

def test_normalize(x, y, expression1):
    symbolic = SymbolicFactor([x, y], expression1)
    sum_variables = symbolic.sum_out_variables([x, y])
    assert sum_variables.expression.sympy_object == 3
    normalized = symbolic.normalize()
    assert normalized.expression.sympy_object == sympy.symbols("x") * sympy.symbols("y") / 3
