from __future__ import annotations

import builtins
from typing import Iterable, List, Set

from .expression import Variable, Expression, Context, Constant
from .basic_expression import BasicExpression
from .z3_expression import Z3SolverExpression


class ClosedInterval(BasicExpression):
    """ [lower_bound, upper_bound] """
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

    def to_range(self) -> Iterable[Expression]:
        """
        If upper and lower bounds are constant, return a range that's iterable.
        Otherwise, raise
        """
        match self.lower_bound, self.upper_bound:
            case Constant(value=l, type=builtins.int), Constant(value=r, type=builtins.int):
                return range(l, r)
            case _:
                raise TypeError("Lower and upper bounds must both be Constants!")

    @property
    def size(self) -> Expression:
        return self.upper_bound - self.lower_bound + 1

    def to_context(self, index: Variable) -> Context:
        result = Z3SolverExpression() & index >= self.lower_bound & index <= self.upper_bound
        assert isinstance(result, Context)  # otherwise lower_bound <= upper_bound is unsatisfiable
        return result


class DottedIntervals(BasicExpression):
    @property
    def subexpressions(self) -> List[Expression]:
        return [self.interval] + self.dots

    def set(self, i: int, new_expression: Expression) -> Expression:
        raise NotImplementedError("TODO")

    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        raise NotImplementedError("TODO")

    def internal_object_eq(self, other) -> bool:
        raise NotImplementedError("TODO")

    def __init__(self, interval: ClosedInterval, dots: List[Expression]):
        super().__init__(Set)
        self._interval = interval
        self._dots = dots

    @property
    def dots(self) -> List[Expression]:
        return self._dots

    @property
    def interval(self) -> ClosedInterval:
        return self._interval

    @property
    def __iter__(self) -> Iterable[Constant]:
        raise NotImplementedError("TODO")


def from_constraints(index: Variable, constraint: Context) -> Expression:
    """
    @param index:
    @param constraint:
    @return: an if-then-else tree whose leaves are DottedIntervals
    """
    raise NotImplementedError("TODO")
