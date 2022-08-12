from .expression import Context, Expression, Constant, AbelianOperation, Variable
from .basic_expression import BasicQuantifierExpression
from .sympy_expression import SymPyExpression
from .interval import ClosedInterval, DottedIntervals, from_constraint
from .util import map_leaves_of_if_then_else
from functools import reduce
from typing import Optional
import operator


def _eliminate_interval(operation: AbelianOperation, index: Variable, interval: ClosedInterval, body: Expression,
                        is_integral: bool) \
        -> Expression:
    # try to solve symbolically
    result = _symbolically_eliminate(operation, index, interval, body, is_integral)
    if result is not None:
        return result

    raise NotImplementedError("sympy cannot eliminate?")
    # TODO: enable this
    # if isinstance(interval.lower_bound, Constant) and isinstance(interval.upper_bound, Constant):
    #     # iterate through the interval if we can
    #     return reduce(operation,
    #                   map(lambda num: body.replace(interval.index, Constant(num)), iter(interval)),
    #                   operation.identity)
    return BasicQuantifierExpression(operation, index, interval.to_context(index), body, is_integral)


def _repeat_n(operation: AbelianOperation, expression: Expression, size: Expression) -> Optional[Expression]:
    if operation.value == operator.add:
        return expression * size
    if operation.value == operator.mul:
        return operator.pow(expression, size)
    return None  # TODO: expand this


def _symbolically_eliminate(operation: AbelianOperation, index: Variable, interval: ClosedInterval, body: Expression,
                            is_integral: bool) \
        -> Optional[Expression]:
    if operation.value == operator.add:
        if not is_integral:
            if (result := Eliminator.symbolic_sum(body, index, interval)) is not None:
                return result
        else:
            if (result := Eliminator.symbolic_integral(body, index, interval)) is not None:
                return result

    if isinstance(body, Constant):
        # repeat addition: multiplication, repeat multiplication: power
        return _repeat_n(operation, body, interval.size)
    return None


class Eliminator:
    integration_counter = 0

    @staticmethod
    def eliminate(operation: AbelianOperation, index: Variable, constraint: Context, body: Expression,
                  is_integral: bool, context: Context) -> Expression:
        def eliminate_at_leaves(dotted_interval: DottedIntervals) -> Expression:
            if not isinstance(dotted_interval, DottedIntervals):
                raise AttributeError("Expect leaves to be DottedIntervals")
            result = _eliminate_interval(operation, index, dotted_interval.interval, body, is_integral)
            if not dotted_interval.dots:  # empty dots
                return result
            inverse = operation.inverse(reduce(operation, dotted_interval.dots, operation.identity))
            return operation(result, inverse)

        try:
            conditional_intervals = from_constraint(index, constraint, context, is_integral)
            print("finished getting intervals")
            return map_leaves_of_if_then_else(conditional_intervals, eliminate_at_leaves)
        except Exception as exc:
            raise AttributeError("disable this for now") from exc
            return BasicQuantifierExpression(operation, index, constraint, body, is_integral)

    @staticmethod
    def symbolic_sum(body: Expression, index: Variable, interval: ClosedInterval) -> Optional[Expression]:
        return SymPyExpression.symbolic_sum(body, index, interval.lower_bound, interval.upper_bound)

    @staticmethod
    def symbolic_integral(body: Expression, index: Variable, interval: ClosedInterval) -> Optional[Expression]:
        if not isinstance(interval.lower_bound, Expression):
            raise NotImplementedError(type(interval.lower_bound))
        if not isinstance(interval.upper_bound, Expression):
            raise NotImplementedError(type(interval.upper_bound))
        Eliminator.integration_counter = Eliminator.integration_counter + 1
        print(f"start {Eliminator.integration_counter}th integration: ({interval.lower_bound},{interval.upper_bound})")
        result = SymPyExpression.symbolic_integral(body, index, interval.lower_bound, interval.upper_bound)
        return result
