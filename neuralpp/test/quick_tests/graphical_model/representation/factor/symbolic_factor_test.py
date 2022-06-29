import pytest
import sympy
from neuralpp.inference.graphical_model.representation.factor.symbolic_factor import SymbolicFactor
from neuralpp.symbolic.sympy_expression import SymPyVariable
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable


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
