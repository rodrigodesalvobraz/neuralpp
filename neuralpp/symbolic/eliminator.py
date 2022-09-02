import operator
from functools import reduce
from typing import Callable, Optional

from .basic_expression import BasicConstant
from .constants import if_then_else
from .expression import Context, Expression, Constant, AbelianOperation, Variable, FunctionApplication
from .interval import ClosedInterval, DottedIntervals, from_constraint
from .profiler import Profiler
from .sympy_expression import SymPyExpression


def _repeat_n(operation: AbelianOperation, expression: Expression, size: Expression) -> Optional[Expression]:
    if operation.value == operator.add:
        return expression * size
    if operation.value == operator.mul:
        return operator.pow(expression, size)
    return None  # TODO: expand this


def _map_leaves_of_if_then_else(conditional_intervals: Expression,
                                function: Callable[[Expression], Expression]) -> Expression:
    """
    @param conditional_intervals: an if-then-else tree
    @param function: map function
    @return: an Expression with each DottedIntervals 'i' mapped to f(i)
    """
    match conditional_intervals:
        case FunctionApplication(function=Constant(value=conditional), arguments=[if_, then, else_]):
            return if_then_else(if_,
                                _map_leaves_of_if_then_else(then, function),
                                _map_leaves_of_if_then_else(else_, function))
        case _:
            return function(conditional_intervals)


class Eliminator:
    def __init__(self, profiler=None):
        if profiler is None:
            self.profiler = Profiler()
        else:
            self.profiler = profiler

    def eliminate(self, operation: AbelianOperation, index: Variable, constraint: Context, body: Expression,
                  is_integral: bool, context: Context) -> Expression:
        def eliminate_at_leaves(dotted_interval: DottedIntervals) -> Expression:
            if not isinstance(dotted_interval, DottedIntervals):
                raise AttributeError("Expect leaves to be DottedIntervals")
            result = self._eliminate_interval(operation, index, dotted_interval.interval, body, is_integral, context)
            if not dotted_interval.dots:  # empty dots
                return result
            inverse = operation.inverse(reduce(operation, dotted_interval.dots, operation.identity))
            return operation(result, inverse)

        try:
            with self.profiler.profile_section("from-constraint"):
                conditional_intervals = from_constraint(index, constraint, context, is_integral, self.profiler)
            return _map_leaves_of_if_then_else(conditional_intervals, eliminate_at_leaves)
        except Exception as exc:
            print(f"cannot eliminate {exc}")
            raise
            return BasicQuantifierExpression(operation, index, constraint, body, is_integral)

    def _eliminate_interval(self, operation: AbelianOperation, index: Variable, interval: ClosedInterval,
                            body: Expression,
                            is_integral: bool, context: Context) \
            -> Expression:
        # try to solve symbolically
        result = self._symbolically_eliminate(operation, index, interval, body, is_integral, context)
        if result is not None:
            return result

        if isinstance(interval.lower_bound, Constant) and isinstance(interval.upper_bound, Constant):
            # iterate through the interval if we can
            return reduce(operation,
                          map(lambda num: body.replace(index, Constant(num)), iter(interval)),
                          operation.identity)
        return BasicQuantifierExpression(operation, index, interval.to_context(index), body, is_integral)

    def _symbolically_eliminate(self, operation: AbelianOperation, index: Variable, interval: ClosedInterval,
                                body: Expression, is_integral: bool, context: Context) -> Optional[Expression]:
        if context.is_known_to_imply(interval.upper_bound <= interval.lower_bound):
            return BasicConstant(0)

        if operation.value == operator.add:
            if not is_integral:
                if (result := Eliminator.symbolic_sum(body, index, interval)) is not None:
                    return result
            else:
                if (result := self.symbolic_integral(body, index, interval)) is not None:
                    if context.is_known_to_imply(interval.upper_bound >= interval.lower_bound):
                        return result
                    return if_then_else(interval.upper_bound > interval.lower_bound, result, 0)

        if isinstance(body, Constant):
            # repeat addition: multiplication, repeat multiplication: power
            return _repeat_n(operation, body, interval.size)
        return None

    @staticmethod
    def symbolic_sum(body: Expression, index: Variable, interval: ClosedInterval) -> Optional[Expression]:
        return SymPyExpression.symbolic_sum(body, index, interval.lower_bound, interval.upper_bound)

    def symbolic_integral(self, body: Expression, index: Variable, interval: ClosedInterval) -> Optional[Expression]:
        if not isinstance(interval.lower_bound, Expression):
            raise NotImplementedError(type(interval.lower_bound))
        if not isinstance(interval.upper_bound, Expression):
            raise NotImplementedError(type(interval.upper_bound))
        with self.profiler.profile_section("integration"):
            if DRY_RUN:
                return BasicConstant(0.0)
            else:
                result = SymPyExpression.symbolic_integral(body, index, interval.lower_bound, interval.upper_bound,
                                                           self.profiler)
                # result = SymPyExpression.symbolic_integral_cached(body, index, interval.lower_bound,
                #                                                   interval.upper_bound, self.profiler)
                return result


# If DRY_RUN flag is set to True, we don't actually do integration, but just return a placeholder for the result.
# The result of `DRY_RUN = True` is not going to be correct, but it gives us a sense of the total number of
# integrations and the run time of other parts of the library quicker.
DRY_RUN = False
