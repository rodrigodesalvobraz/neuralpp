from .expression import QuantifierExpression, Context, Expression, Constant, AbelianOperation, Variable, \
    FunctionApplication
from .basic_expression import BasicQuantifierExpression
from .sympy_expression import SymPyExpression
from .interval import ClosedInterval, DottedIntervals, from_constraint
from .constants import if_then_else
import neuralpp.symbolic.functions as functions
from functools import reduce
from typing import Optional, Callable
import operator


def _eliminate_interval(operation: AbelianOperation, index: Variable, interval: ClosedInterval, body: Expression) \
        -> Expression:
    # try to solve symbolically
    result = _symbolically_eliminate(operation, index, interval, body)
    if result is not None:
        return result

    if isinstance(interval.lower_bound, Constant) and isinstance(interval.upper_bound, Constant):
        # iterate through the interval if we can
        return reduce(operation,
                      map(lambda num: body.replace(interval.index, Constant(num)), iter(interval)),
                      operation.identity)

    return BasicQuantifierExpression(operation, index, interval.to_context(index), body)


def _repeat_n(operation: AbelianOperation, expression: Expression, size: Expression) -> Optional[Expression]:
    if operation.value == operator.add:
        return expression * size
    if operation.value == operator.mul:
        return operator.pow(expression, size)
    return None  # TODO: expand this


def _symbolically_eliminate(operation: AbelianOperation, index: Variable, interval: ClosedInterval, body: Expression) \
        -> Optional[Expression]:
    if operation.value == operator.add:
        if (result := Eliminator.symbolic_sum(body, index, interval)) is not None:
            return result

    if isinstance(body, Constant):
        # repeat addition: multiplication, repeat multiplication: power
        return _repeat_n(operation, body, interval.size)
    return None


def _map_leaves(conditional_intervals: Expression,
                function: Callable[[DottedIntervals], Expression]) -> Expression:
    """
    @param conditional_intervals: expect an if-then-else tree with leaves being DottedIntervals
    @param function: map function
    @return: an Expression with each DottedIntervals i mapped to f(i)
    """
    match conditional_intervals:
        case FunctionApplication(function=Constant(value=functions.conditional), arguments=[if_, then, else_]):
            return if_then_else(if_, _map_leaves(then, function), _map_leaves(else_, function))
        case DottedIntervals():
            return function(conditional_intervals)
        case _:
            raise AttributeError("Unexpected subtree.")


class Eliminator:
    @staticmethod
    def eliminate(operation: AbelianOperation, index: Variable, constraint: Context, body: Expression,
                  context: Context) -> Expression:
        def eliminate_at_leaves(dotted_interval: DottedIntervals) -> Expression:
            result = _eliminate_interval(operation, index, dotted_interval.interval, body)
            if not dotted_interval.dots:  # empty dots
                return result
            inverse = operation.inverse(reduce(operation, dotted_interval.dots, operation.identity))
            return operation(result, inverse)

        try:
            conditional_intervals = from_constraint(index, constraint & context)
            return _map_leaves(conditional_intervals, eliminate_at_leaves)
        except Exception:
            return BasicQuantifierExpression(operation, index, constraint, body)

    @staticmethod
    def symbolic_sum(body: Expression, index: Variable, interval: ClosedInterval) -> Optional[Expression]:
        return SymPyExpression.symbolic_sum(body, index, interval.lower_bound, interval.upper_bound)
