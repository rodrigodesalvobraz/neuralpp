import sympy
from neuralpp.inference.graphical_model.representation.factor.symbolic_factor import (
    SymbolicFactor,
)
from neuralpp.symbolic.sympy_expression import (
    SymPyVariable,
    SymPyConstant,
    SymPyFunctionApplication,
)
from neuralpp.inference.graphical_model.variable.integer_variable import (
    IntegerVariable,
)
from neuralpp.symbolic.constants import if_then_else
from neuralpp.symbolic.sympy_interpreter import SymPyInterpreter
from neuralpp.inference.graphical_model.variable_elimination import (
    VariableElimination,
)
from neuralpp.inference.graphical_model.brute_force import BruteForce
from neuralpp.symbolic.general_normalizer import GeneralNormalizer
from neuralpp.symbolic.z3_expression import Z3SolverExpression


def test_sympy_condition():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = x_sympy * y_sympy

    symbolic = SymbolicFactor([x, y], expression1)
    conditioned = symbolic.condition({x: 1})

    expected = SymPyVariable(y_symbol, int)
    assert conditioned.expression.syntactic_eq(expected)


def test_sympy_if_then_else_condition():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = if_then_else(x_sympy > y_sympy, 2, 3)

    symbolic = SymbolicFactor([x, y], expression1)
    conditioned1 = symbolic.condition({x: 1})

    expected1 = if_then_else(y_sympy < 1, 2, 3)
    # TODO: Fix this test
    # assert conditioned1.expression.syntactic_eq(expected1)

    conditioned2 = symbolic.condition({x: 1, y: 2})
    expected2 = SymPyConstant.new_constant(3)
    assert conditioned2.expression.syntactic_eq(expected2)

    conditioned3 = symbolic.condition({x: 1, y: 0})
    expected3 = SymPyConstant.new_constant(2)
    assert conditioned3.expression.syntactic_eq(expected3)


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
    symbolic3 = symbolic1 * symbolic2

    symbolic1_condition = symbolic1.condition({x: 1, y: 2})
    symbolic2_condition = symbolic2.condition({x: 1, z: 3})
    symbolic3_condition = symbolic3.condition({x: 1, y: 2, z: 3})

    expected = symbolic1_condition.expression * symbolic2_condition.expression
    expected = SymPyInterpreter().simplify(expected)
    assert symbolic3_condition.expression.syntactic_eq(expected)


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
    symbolic3 = symbolic1 * symbolic2

    symbolic1_condition = symbolic1.condition({x: 3, y: 2})
    symbolic2_condition = symbolic2.condition({x: 3, z: 3})
    symbolic3_condition = symbolic3.condition({x: 3, y: 2, z: 3})

    expected = symbolic1_condition.expression * symbolic2_condition.expression
    expected = SymPyInterpreter().simplify(expected)
    assert symbolic3_condition.expression.syntactic_eq(expected)


def test_sum_out_variable():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = x_sympy * y_sympy

    symbolic = SymbolicFactor([x, y], expression1)
    sum_out_x = symbolic ^ x

    expected = SymPyFunctionApplication(3 * y_symbol, {y_symbol: int})
    assert sum_out_x.expression.syntactic_eq(expected)


def test_if_then_else_sum_out_variable():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = if_then_else(x_sympy > y_sympy, 2, 3)

    symbolic = SymbolicFactor([x, y], expression1)
    sum_out_x = symbolic ^ x

    expected = (
        if_then_else(0 > y_sympy, 2, 3)
        + if_then_else(1 > y_sympy, 2, 3)
        + if_then_else(2 > y_sympy, 2, 3)
    )
    expected = SymPyInterpreter().simplify(expected)
    assert sum_out_x.expression.syntactic_eq(expected)


def test_normalize():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = x_sympy * y_sympy

    symbolic = SymbolicFactor([x, y], expression1)
    sum_variables = symbolic.sum_out_variables([x, y])

    expected1 = SymPyConstant.new_constant(3)
    assert sum_variables.expression.syntactic_eq(expected1)

    normalized = symbolic.normalize()
    expected2 = SymPyFunctionApplication(
        x_symbol * y_symbol / 3, {x_symbol: int, y_symbol: int}
    )
    assert normalized.expression.syntactic_eq(expected2)


def test_if_then_else_normalize():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)

    x_symbol, y_symbol = sympy.symbols("x y")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)

    expression1 = if_then_else(x_sympy > y_sympy, 2, 3)

    symbolic = SymbolicFactor([x, y], expression1)
    sum_variables = symbolic.sum_out_variables([x, y])

    expected1 = SymPyConstant.new_constant(15)
    assert sum_variables.expression.syntactic_eq(expected1)

    normalized = symbolic.normalize()
    expected2 = SymPyInterpreter().simplify(if_then_else(x_sympy > y_sympy, 2, 3) / 15)
    assert normalized.expression.syntactic_eq(expected2)


def test_with_variable_elimination():
    x = IntegerVariable("x", 3)
    y = IntegerVariable("y", 2)
    z = IntegerVariable("z", 2)

    x_symbol, y_symbol, z_symbol = sympy.symbols("x y z")
    x_sympy = SymPyVariable(x_symbol, int)
    y_sympy = SymPyVariable(y_symbol, int)
    z_sympy = SymPyVariable(z_symbol, int)

    model = [
        SymbolicFactor(
            [x, y],
            if_then_else(x_sympy == 2, 0.5, if_then_else(x_sympy == y_sympy, 1.0, 0.0)),
        ),
        SymbolicFactor([y, z], if_then_else(y_sympy == z_sympy, 1.0, 0.0)),
    ]

    query = z

    ve_result = VariableElimination().run(query, model)
    brute_result = BruteForce().run(query, model)

    difference = ve_result.expression - brute_result.expression
    normalizer = GeneralNormalizer()
    context = Z3SolverExpression()

    expected = SymPyConstant.new_constant(0)
    assert normalizer.normalize(difference, context).syntactic_eq(expected)
