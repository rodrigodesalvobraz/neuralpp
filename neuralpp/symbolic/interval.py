from __future__ import annotations

import builtins
import operator
from sympy import solve, oo
from typing import Iterable, List, Set, Optional
from .expression import Variable, Expression, Context, Constant, FunctionApplication
from .basic_expression import BasicExpression
from .z3_expression import Z3SolverExpression
from .sympy_interpreter import SymPyInterpreter
from .sympy_expression import SymPyExpression
from .constants import min_, max_
from .profiler import Profiler

_simplifier = SymPyInterpreter()


class ClosedInterval(BasicExpression):
    """ [lower_bound, upper_bound] """
    def __init__(self, lower_bound, upper_bound):
        if not isinstance(lower_bound, Expression):
            raise AttributeError(f"{lower_bound}")
        if not isinstance(upper_bound, Expression):
            raise AttributeError(f"{upper_bound}")
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


def _adjust(expression: Expression, variable: Variable) -> Expression:
    sympy_expression = SymPyExpression.convert(expression)
    sympy_var = SymPyExpression.convert(variable).sympy_object
    answer = solve(sympy_expression.sympy_object, sympy_var)
    if len(answer.args) > 2:
        raise AttributeError("?")
    if len(answer.args) == 2:
        if answer.args[0].has(oo) or answer.args[0].has(-oo):
            answer = answer.args[1]
        elif answer.args[1].has(oo) or answer.args[1].has(-oo):
            answer = answer.args[0]
    if answer.has(oo) or answer.has(-oo):
        raise NotImplementedError(f"TODO {answer} {sympy_expression.sympy_object}")
    return SymPyExpression.from_sympy_object(answer, sympy_expression.type_dict)


class MagicInterval:
    """
    One interval, but with non-deterministic lower/upperbounds, for example,
    MagicInterval [{x,y}, {z,y}]  is
    [x,z] if x > y and z <= y
    [y,z] if x <= y and z > y
    [x,y] if x > y and z >= y
    [y,y] if x <= y and z <= y
    """
    def __init__(self):
        self._lower_bounds = []
        self._upper_bounds = []

    @property
    def lower_bounds(self):
        return self._lower_bounds

    @property
    def upper_bounds(self):
        return self._upper_bounds

    def add_lower_bound(self, lower_bound: Expression):
        self._lower_bounds.append(lower_bound)

    def add_upper_bound(self, upper_bound: Expression):
        self._upper_bounds.append(upper_bound)

    def to_conditional_intervals(self, context: Z3SolverExpression) -> Expression:
        if len(self.lower_bounds) < 1 or len(self.upper_bounds) < 1:
            raise AttributeError(f"bounds not set. {self.lower_bounds} {self.upper_bounds}")
        return DottedIntervals(ClosedInterval(max_(self._lower_bounds),
                                              min_(self.upper_bounds)), [])


def from_constraint(index: Variable, constraint: Context, context: Context, is_integral: bool,
                    profiler: Optional[Profiler] = None,
                    ) -> Expression:
    """
    @param index: the variable that the interval is for
    @param constraint: the constraint of the quantifier expression
    @param context: the context that the expression is in
    @param is_integral: whether asking for an integration (if yes return as is, instead of rounding), a bit hacky
    @param profiler: optional profiler
    @return: an DottedInterval

    This currently only supports the most basic of constraints
    For example, x > 0 and x <= 5 should return an interval [1, 5]
    More complicated cases will be added later
    """
    with profiler.profile_section("to-dnf"):
        constraint = SymPyExpression.convert(constraint)
        # constraint = SymPyInterpreter().simplify(constraint)  # get an DNF
    match constraint:
        case FunctionApplication(function=Constant(value=operator.or_)):
            raise NotImplementedError("Not expecting OR")
        case FunctionApplication(function=Constant(value=operator.and_), arguments=arguments):
            with profiler.profile_section("compute magic interval"):
                magic_interval = MagicInterval()
                for argument in arguments:
                    argument = _adjust(argument, index)
                    _extract_bound_from_constraint(index, argument, magic_interval, is_integral)
                return magic_interval.to_conditional_intervals(context)
        case _:
            raise NotImplementedError("Constraint should be AND of constraints")


def _extract_bound_from_constraint(
    index: Variable,
    constraint: Expression,
    magic_interval: MagicInterval,
    is_integral: bool,
):
    """
    @param index: the variable that the interval is for
    @param constraint: the context that constrains the variable
    @param magic_interval: record tracker
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
                _check_and_set_bounds(0, bound, magic_interval)
            elif variable_index == 2:
                _check_and_set_bounds(1, bound, magic_interval)
        case operator.le:
            if variable_index == 1:
                _check_and_set_bounds(1, bound, magic_interval)
            elif variable_index == 2:
                _check_and_set_bounds(0, bound, magic_interval)
        case operator.gt:
            if variable_index == 1:
                if not is_integral:
                    bound = _simplifier.simplify(bound + 1)
                _check_and_set_bounds(0, bound, magic_interval)
            elif variable_index == 2:
                if not is_integral:
                    bound = _simplifier.simplify(bound - 1)
                _check_and_set_bounds(1, bound, magic_interval)
        case operator.lt:
            if variable_index == 1:
                if not is_integral:
                    bound = _simplifier.simplify(bound - 1)
                _check_and_set_bounds(1, bound, magic_interval)
            elif variable_index == 2:
                if not is_integral:
                    bound = _simplifier.simplify(bound + 1)
                _check_and_set_bounds(0, bound, magic_interval)
        case _:
            raise ValueError(f"interval doesn't support {possible_inequality} yet")
    return magic_interval


def _check_and_set_bounds(
    index: int,
    bound: Expression,
    magic_interval: MagicInterval
):
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
    match index:
        case 0:
            magic_interval.add_lower_bound(bound)
        case 1:
            magic_interval.add_upper_bound(bound)
        case _:
            raise IndexError(f"{index} is out of bounds")
