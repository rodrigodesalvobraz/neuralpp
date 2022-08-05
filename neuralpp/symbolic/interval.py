from __future__ import annotations
from abc import abstractmethod
from typing import Iterable, Callable
from constants import if_then_else

from .expression import Variable, Expression, Context
from .z3_expression import Z3SolverExpression


class ClosedInterval:
    """ [lower_bound, upper_bound] """
    @property
    @abstractmethod
    def lower_bound(self) -> Expression:
        pass

    @property
    @abstractmethod
    def upper_bound(self) -> Expression:
        pass

    @abstractmethod
    def to_range(self) -> Iterable[Expression]:
        """ if upper and lower bounds are constant, return a range that's iterable """
        pass

    @property
    def length(self) -> Expression:
        return self.upper_bound - self.lower_bound

    def to_context(self, index: Variable) -> Context:
        result = Z3SolverExpression() & index >= self.lower_bound & index <= self.upper_bound
        assert isinstance(result, Context)  # otherwise lower_bound <= upper_bound is unsatisfiable
        return result


class GeneralIntervals:
    @staticmethod
    def from_constraint(index: Variable, constraint: Expression) -> GeneralIntervals:
        pass

    @abstractmethod
    def map_leaves(self, function: Callable[[DottedIntervals], Expression]) -> Expression:
        """ Recurse over the leaves (DottedIntervals), apply @param `function` to them, and compose an Expression. """
        pass


class DottedIntervals(GeneralIntervals):
    @property
    @abstractmethod
    def __iter__(self) -> Iterable[ClosedInterval]:
        pass

    def map_leaves(self, function: Callable[[DottedIntervals], Expression]) -> Expression:
        return function(self)


class ConditionalIntervals(GeneralIntervals):
    @property
    @abstractmethod
    def if_(self) -> Expression:
        pass

    @property
    @abstractmethod
    def then(self) -> GeneralIntervals:
        pass

    @property
    @abstractmethod
    def else_(self) -> GeneralIntervals:
        pass

    def map_leaves(self, function: Callable[[DottedIntervals], Expression]) -> Expression:
        return if_then_else(self.if_, self.then.map_leaves(function), self.else_.map_leaves(function))
