from __future__ import annotations

import builtins
import operator
from typing import Iterable, List, Set, Tuple, Optional
from .expression import Variable, Expression, Context, Constant, FunctionApplication, QuantifierExpression
from .basic_expression import BasicExpression
from .z3_expression import Z3SolverExpression, Z3Expression
from .sympy_interpreter import SymPyInterpreter

_simplifier = SymPyInterpreter()


class ClosedInterval(BasicExpression):
    """ [lower_bound, upper_bound] """
    def __init__(self, lower_bound, upper_bound):
        super().__init__(Set)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    @property
    def lower_bound(self) -> Expression:
        return self._lower_bound

    @property
    def upper_bound(self) -> Expression:
        return self._upper_bound

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.lower_bound, self.upper_bound]

    def set(self, i: int, new_expression: Expression) -> Expression:
        if i == 0:
            return ClosedInterval(new_expression, self.upper_bound)
        if i == 1:
            return ClosedInterval(self.lower_bound, new_expression)
        raise IndexError("out of scope.")

    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        if self.syntactic_eq(from_expression):
            return to_expression
        return ClosedInterval(self.lower_bound.replace(from_expression, to_expression),
                              self.upper_bound.replace(from_expression, to_expression))

    def internal_object_eq(self, other) -> bool:
        if not isinstance(other, ClosedInterval):
            return False
        return self.lower_bound.internal_object_eq(other.lower_bound) and \
            self.upper_bound.internal_object_eq(other.upper_bound)

    def __iter__(self) -> Iterable[int]:
        """
        If upper and lower bounds are constant, return a range that's iterable.
        Otherwise, raise
        """
        match self.lower_bound, self.upper_bound:
            case Constant(value=l, type=builtins.int), Constant(value=r, type=builtins.int):
                return iter(range(l, r))
            case _:
                raise TypeError("Lower and upper bounds must both be Constants!")

    @property
    def size(self) -> Expression:
        if self.lower_bound > self.upper_bound:
            raise AttributeError(f'[{self.lower_bound},{self.upper_bound}] is an empty interval.')
        return self.upper_bound - self.lower_bound + 1

    def to_context(self, index: Variable) -> Context:
        if self.lower_bound is None:
            raise AttributeError("lower bound is None")
        if self.upper_bound is None:
            raise AttributeError("upper bound is None")
        result = Z3SolverExpression() & (index >= self.lower_bound) & (index <= self.upper_bound)
        assert isinstance(result, Context)  # otherwise lower_bound <= upper_bound is unsatisfiable
        return result


class DottedIntervals(BasicExpression):
    def __init__(self, interval: ClosedInterval, dots: List[Expression]):
        super().__init__(Set)
        self._interval = interval
        self._dots = dots

    @property
    def interval(self) -> ClosedInterval:
        return self._interval

    @property
    def dots(self) -> List[Expression]:
        return self._dots

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.interval] + self.dots

    def set(self, i: int, new_expression: Expression) -> Expression:
        raise NotImplementedError("TODO")

    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        raise NotImplementedError("TODO")

    def internal_object_eq(self, other) -> bool:
        raise NotImplementedError("TODO")

    @property
    def __iter__(self) -> Iterable[int]:
        raise NotImplementedError("TODO")


def from_constraint(index: Variable, constraint: Context, context: Context, is_integral: bool) -> Expression:
    """
    @param index: the variable that the interval is for
    @param constraint: the constraint of the quantifier expression
    @param context: the context that the expression is in
    @param is_integral: whether asking for an integration (if yes return as is, instead of rounding), a bit hacky
    @return: an DottedInterval

    This currently only supports the most basic of constraints
    For example, x > 0 and x <= 5 should return an interval [1, 5]
    More complicated cases will be added later
    """
    closed_interval = ClosedInterval(None, None)
    exceptions = []
    for subexpression in constraint.subexpressions:
        if isinstance(subexpression, FunctionApplication):
            closed_interval, exceptions = _extract_bound_from_constraint(index, subexpression, closed_interval, exceptions, is_integral)

    if isinstance(context, QuantifierExpression):
        context_interval = from_constraint(context.index, context.constraint, None, context.is_integral)
        context_lower_bound = context_interval.interval.lower_bound
        context_upper_bound = context_interval.interval.upper_bound
        if (context_lower_bound.contains(index)):
            bound = _simplifier.simplify(context_upper_bound - context_lower_bound + index)
            closed_interval = _check_and_set_bounds(1, bound, closed_interval)

        elif (context_upper_bound.contains(index)):
            bound = _simplifier.simplify(context_lower_bound - context_upper_bound + index)
            closed_interval = _check_and_set_bounds(0, bound, closed_interval)

    return DottedIntervals(closed_interval, exceptions)


def _extract_bound_from_constraint(
    index: Variable,
    constraint: Expression,
    closed_interval: ClosedInterval,
    exceptions: List[Expression],
    is_integral: bool,
) -> Tuple[ClosedInterval, List[Expression]]:
    """
    @param index: the variable that the interval is for
    @param constraint: the context that constrains the variable
    @param closed_interval: the current ClosedInterval
    @param exceptions: a list of exceptions
    @param is_integral: whether we're extracting bound for an integral (in which case don't round)
    @return: a tuple of closed_interval and list of exceptions

    Extracts the operator, where in the expression the variable is, and possible lower or upper bound

    For example, the constraint is (>=, x, 5). Possible_inequality will be >=, variable_index will be 1,
    and bound will be 5

    Sets the possible bound if it's greater than the current lower or less than the current upper by passing
    it into _check_and_set_bounds
    """
    possible_inequality = constraint.subexpressions[0].value

    variable_index = None
    bound = None
    if constraint.subexpressions[1].syntactic_eq(index):
        variable_index = 1
        bound = constraint.subexpressions[2]
    elif constraint.subexpressions[2].syntactic_eq(index):
        variable_index = 2
        bound = constraint.subexpressions[1]
    else:
        raise ValueError(f"intervals is not yet ready to handle more complicated cases {constraint} {index}")

    match possible_inequality:
        case operator.ge:
            if variable_index == 1:
                closed_interval = _check_and_set_bounds(0, bound, closed_interval)
            elif variable_index == 2:
                closed_interval = _check_and_set_bounds(1, bound, closed_interval)
        case operator.le:
            if variable_index == 1:
                closed_interval = _check_and_set_bounds(1, bound, closed_interval)
            elif variable_index == 2:
                closed_interval = _check_and_set_bounds(0, bound, closed_interval)
        case operator.gt:
            if variable_index == 1:
                if not is_integral:
                    bound = _simplifier.simplify(bound + 1)
                closed_interval = _check_and_set_bounds(0, bound, closed_interval)
            elif variable_index == 2:
                if not is_integral:
                    bound = _simplifier.simplify(bound - 1)
                closed_interval = _check_and_set_bounds(1, bound, closed_interval)
        case operator.lt:
            if variable_index == 1:
                if not is_integral:
                    bound = _simplifier.simplify(bound - 1)
                closed_interval = _check_and_set_bounds(1, bound, closed_interval)
            elif variable_index == 2:
                if not is_integral:
                    bound = _simplifier.simplify(bound + 1)
                closed_interval = _check_and_set_bounds(0, bound, closed_interval)
        case operator.eq:
            closed_interval = _check_and_set_bounds(0, bound, closed_interval)
            closed_interval = _check_and_set_bounds(1, bound, closed_interval)
        case operator.ne:
            exceptions.append(bound)

        case _:
            raise ValueError(f"interval doesn't support {possible_inequality} yet")
    return closed_interval, exceptions


def _check_and_set_bounds(
    index: int,
    bound: Expression,
    closed_interval: ClosedInterval
) -> Tuple[ClosedInterval, List[Expression]]:
    """
    @param index: indicates which bound we are checking => 0 is lower bound and 1 is upper bound
    @param bound: the Constant of the bound (the value inside will be an int)
    @param closed_interval: the current ClosedInterval we have
    @exceptions: a list of exceptions
    @return: a tuple of the new closed_interval and list of exceptions

    When checking the lower bounds: if the bound is >= the current lower bound, replace the current
    lower bound with the bound. For example, if the context is x >= 4 and x >= 5, x has to be greater than 5
    to fulfill the context.

    When checking the upper bounds: if the bound is <= the current lower bound, replace the current
    upper bound with the bound.
    """
    lower_bound = closed_interval.lower_bound
    upper_bound = closed_interval.upper_bound
    match index:
        case 0:
            if lower_bound is None:
                lower_bound = bound
                closed_interval = closed_interval.set(0, bound)
            elif isinstance(lower_bound, Constant) and isinstance(bound, Constant) and bound.value >= lower_bound.value:
                lower_bound = bound
                closed_interval = closed_interval.set(0, bound)
        case 1:
            if upper_bound is None:
                closed_interval = closed_interval.set(1, bound)
            elif isinstance(upper_bound, Constant) and isinstance(bound, Constant) and bound.value <= upper_bound.value:
                upper_bound = bound
                closed_interval = closed_interval.set(1, bound)
        case _:
            raise IndexError(f"{index} is out of bounds")

    if isinstance(lower_bound, Constant) and isinstance(upper_bound, Constant):
        if lower_bound.value > upper_bound.value:
            raise ValueError(f"the lower bound ({lower_bound}) is greater than upper bound ({upper_bound})")

    return closed_interval
