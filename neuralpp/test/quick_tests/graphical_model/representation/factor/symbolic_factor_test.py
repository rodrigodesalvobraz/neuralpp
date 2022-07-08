import pytest
import sympy
import operator
from neuralpp.inference.graphical_model.representation.factor.symbolic_factor import SymbolicFactor
from neuralpp.symbolic.sympy_expression import SymPyVariable
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.symbolic.constants import if_then_else
from neuralpp.symbolic.sympy_interpreter import SymPyInterpreter

def test_sympy_condition():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = x_sympy * y_sympy

    symbolic = SymbolicFactor([x, y], expression1)
    conditioned = symbolic.condition({x: 1})
    assert conditioned.expression == SymPyVariable(y_symbol, int)

def test_sympy_if_then_else_condition():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = if_then_else(x_sympy > y_sympy, 2, 3)

    symbolic = SymbolicFactor([x, y], expression1)
    conditioned1 = symbolic.condition({x: 1})
    assert conditioned1.expression.arguments[0].syntactic_eq(y_sympy < 1)
    assert conditioned1.expression.arguments[1].value == 2
    assert conditioned1.expression.arguments[2].value == 3

    conditioned1 = symbolic.condition({x: 1, y:2})
    assert conditioned1.expression == 3

    conditioned1 = symbolic.condition({x: 1, y:0})
    assert conditioned1.expression == 2

def test_mul_by_non_identity():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)
    z = IntegerVariable("z", 3)

    x_symbol, y_symbol, z_symbol = sympy.symbols("x y z")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)
    z_sympy = SymPyVariable(z_symbol, int)

    expression1 = x_sympy * y_sympy
    expression2 = x_sympy + z_sympy

    symbolic1 = SymbolicFactor([x, y], expression1)
    symbolic2 = SymbolicFactor([x, z], expression2)
    symbolic3 = symbolic1.mul_by_non_identity(symbolic2)

    symbolic1_condition = symbolic1.condition({x: 1, y: 2})
    symbolic2_condition = symbolic2.condition({x: 1, z: 3})
    symbolic3_condition = symbolic3.condition({x: 1, y: 2, z: 3})

    assert symbolic1_condition.expression.sympy_object * symbolic2_condition.expression.sympy_object == symbolic3_condition.expression.sympy_object


def test_if_then_else_mul_by_non_identity():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)
    z = IntegerVariable("z", 3)

    x_symbol, y_symbol, z_symbol = sympy.symbols("x y z")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)
    z_sympy = SymPyVariable(z_symbol, int)

    expression1 = if_then_else(x_sympy > y_sympy, 2, 3)
    expression2 = if_then_else(x_sympy == z_sympy, 1, 5)

    symbolic1 = SymbolicFactor([x, y], expression1)
    symbolic2 = SymbolicFactor([x, z], expression2)
    symbolic3 = symbolic1.mul_by_non_identity(symbolic2)

    symbolic1_condition = symbolic1.condition({x: 3, y: 2})
    symbolic2_condition = symbolic2.condition({x: 3, z: 3})
    symbolic3_condition = symbolic3.condition({x: 3, y: 2, z: 3})

    assert symbolic1_condition.expression.sympy_object * symbolic2_condition.expression.sympy_object == symbolic3_condition.expression.sympy_object

def test_sum_out_variable():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = x_sympy * y_sympy

    symbolic = SymbolicFactor([x, y], expression1)
    sum_out_x = symbolic.sum_out_variable(x)

    assert sum_out_x.expression.sympy_object == 3 * y_symbol

def test_if_then_else_sum_out_variable():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = if_then_else(x_sympy > y_sympy, 2, 3)

    symbolic = SymbolicFactor([x, y], expression1)
    sum_out_x = symbolic.sum_out_variable(x)
    result = if_then_else(0 > y_sympy, 2, 3) + if_then_else(1 > y_sympy, 2, 3) + if_then_else(2 > y_sympy, 2, 3)
    result = SymPyInterpreter().simplify(result)

    assert sum_out_x.expression.syntactic_eq(result)

def test_normalize():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = x_sympy * y_sympy

    symbolic = SymbolicFactor([x, y], expression1)
    sum_variables = symbolic.sum_out_variables([x, y])

    assert sum_variables.expression.sympy_object == 3

    normalized = symbolic.normalize()

    assert normalized.expression.sympy_object == x_symbol * y_symbol / 3

def test_if_then_else_normalize():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = if_then_else(x_sympy > y_sympy, 2, 3)

    symbolic = SymbolicFactor([x, y], expression1)
    sum_variables = symbolic.sum_out_variables([x, y])

    assert sum_variables.expression.sympy_object == 15

    normalized = symbolic.normalize()
    result = SymPyInterpreter().simplify(if_then_else(x_sympy > y_sympy, 2, 3) / 15)

    assert normalized.expression.syntactic_eq(result)
