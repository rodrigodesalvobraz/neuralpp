import pytest
from neuralpp.symbolic.interval import from_constraint
from neuralpp.symbolic.z3_expression import Z3SolverExpression, Z3Expression
from neuralpp.symbolic.basic_expression import BasicVariable, BasicSummation
from neuralpp.symbolic.expression import FunctionApplication


def test_basic_constant_intervals():
    i = BasicVariable('i', int)

    empty_context = Z3SolverExpression()
    constant_context = empty_context & (i < 5) & (i > 0)

    dotted_interval = from_constraint(i, constant_context, empty_context, False)
    interval = dotted_interval.interval

    assert interval.lower_bound.syntactic_eq(Z3Expression.new_constant(1))
    assert interval.upper_bound.syntactic_eq(Z3Expression.new_constant(4))


def test_basic_symbolic_intervals():
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

def test_intervals_with_summation():
    i = BasicVariable('i', int)
    j = BasicVariable('j', int)

    empty_context = Z3SolverExpression()
    constant_context = empty_context & (j < 20) & (j > 0)

    sum_upper_bound = BasicSummation(int, i, Z3SolverExpression.from_expression(j < i) & (i < 10), i + j)

    dotted_interval = from_constraint(j, constant_context, sum_upper_bound, False)
    interval = dotted_interval.interval

    assert interval.lower_bound.syntactic_eq(Z3Expression.new_constant(1))
    assert interval.upper_bound.syntactic_eq(Z3Expression.new_constant(8))


    sum_lower_bound = BasicSummation(int, i, Z3SolverExpression.from_expression(j > i) & (i > 10), i + j)

    dotted_interval = from_constraint(j, constant_context, sum_lower_bound, False)
    interval = dotted_interval.interval

    assert interval.lower_bound.syntactic_eq(Z3Expression.new_constant(12))
    assert interval.upper_bound.syntactic_eq(Z3Expression.new_constant(19))

def test_invalid_constant_intervals():
    i = BasicVariable('i', int)
    j = BasicVariable('j', int)

    empty_context = Z3SolverExpression()
    constant_context = empty_context & (j < 20) & (j > 0)

    sum_invalid = BasicSummation(int, i, Z3SolverExpression.from_expression(j < i) & (i < 0), i + j)

    with pytest.raises(ValueError):
        dotted_interval = from_constraint(i, constant_context, empty_context, False)
