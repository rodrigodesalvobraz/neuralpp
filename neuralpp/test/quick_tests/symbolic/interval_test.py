from neuralpp.symbolic.interval import from_constraint
from neuralpp.symbolic.z3_expression import Z3SolverExpression, Z3Expression
from neuralpp.symbolic.basic_expression import BasicVariable, BasicSummation
from neuralpp.symbolic.expression import FunctionApplication


def test_basic_constant_closed_intervals():
    i = BasicVariable('i', int)

    empty_context = Z3SolverExpression()
    constant_context = empty_context & (i < 5) & (i > 0)

    dotted_interval = from_constraint(i, constant_context, empty_context, False)
    interval = dotted_interval.interval

    assert interval.lower_bound.syntactic_eq(Z3Expression.new_constant(1))
    assert interval.upper_bound.syntactic_eq(Z3Expression.new_constant(4))


def test_basic_symbolic_closed_intervals():
    i = BasicVariable('i', int)
    x = BasicVariable('x', int)

    empty_context = Z3SolverExpression()
    constant_context = empty_context & (i < 5) & (i > x)

    dotted_interval = from_constraint(i, constant_context, empty_context, False)
    interval = dotted_interval.interval

    assert interval.lower_bound.syntactic_eq(1 + x)
    assert interval.upper_bound.syntactic_eq(Z3Expression.new_constant(4))

    empty_context = Z3SolverExpression()
    constant_context = empty_context & (i <= x + 5) & (i > x + 2)

    dotted_interval = from_constraint(i, constant_context, empty_context, False)
    interval = dotted_interval.interval

    assert interval.lower_bound.syntactic_eq(3 + x)
    assert interval.upper_bound.syntactic_eq(x + 5)
