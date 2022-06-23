import pytest

import sympy
from typing import List, Any, Optional, Type, Callable, Dict
from neuralpp.inference.graphical_model.representation.factor.symbolic_factor import SymbolicFactor
from neuralpp.symbolic.sympy_expression import SymPyVariable, SymPyFunctionApplication
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable

int_to_int_to_int = Callable[[int, int], int]

def test_sympy_condition():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_sympy, y_sympy = sympy.symbols("x y")
    expression1 = SymPyFunctionApplication(x_sympy * y_sympy, {x_sympy: int, y_sympy: int}, int_to_int_to_int)

    symbolic = SymbolicFactor([x, y], expression1)
    conditioned = symbolic.condition({x: 1})
    assert conditioned.expression == SymPyVariable(y_sympy, int)

def test_mul_by_non_identity():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)
    z = IntegerVariable("z", 3)

    x_sympy, y_sympy, z_sympy = sympy.symbols("x y z")
    expression1 = SymPyFunctionApplication(x_sympy * y_sympy, {x_sympy: int, y_sympy: int}, int_to_int_to_int)
    expression2 = SymPyFunctionApplication(x_sympy + z_sympy, {x_sympy: int, z_sympy: int}, int_to_int_to_int)

    symbolic1 = SymbolicFactor([x, y], expression1)
    symbolic2 = SymbolicFactor([x, z], expression2)
    symbolic3 = symbolic1.mul_by_non_identity(symbolic2)

    symbolic1_condition = symbolic1.condition({x: 1, y: 2})
    symbolic2_condition = symbolic2.condition({x: 1, z: 3})
    symbolic3_condition = symbolic3.condition({x: 1, y: 2, z: 3})

    assert symbolic1_condition.expression.sympy_object * symbolic2_condition.expression.sympy_object == symbolic3_condition.expression.sympy_object

def test_sum_out_variable():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_sympy, y_sympy = sympy.symbols("x y")
    expression1 = SymPyFunctionApplication(x_sympy * y_sympy, {x_sympy: int, y_sympy: int}, int_to_int_to_int)

    symbolic = SymbolicFactor([x, y], expression1)
    sum_out_x = symbolic.sum_out_variable(x)
    assert sum_out_x.expression.sympy_object == 3 * y_sympy

def test_normalize():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_sympy, y_sympy = sympy.symbols("x y")
    expression1 = SymPyFunctionApplication(x_sympy * y_sympy, {x_sympy: int, y_sympy: int}, int_to_int_to_int)

    symbolic = SymbolicFactor([x, y], expression1)
    sum_variables = symbolic.sum_out_variables([x, y])
    assert sum_variables.expression.sympy_object == 3
    normalized = symbolic.normalize()
    assert normalized.expression.sympy_object == x_sympy * y_sympy / 3
