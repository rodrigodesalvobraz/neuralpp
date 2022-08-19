from .expression import Context, Expression, Constant, AbelianOperation, Variable
from .basic_expression import BasicQuantifierExpression, BasicConstant
from .sympy_expression import SymPyExpression
from .interval import ClosedInterval, DottedIntervals, from_constraint
from .util import map_leaves_of_if_then_else
from .constants import if_then_else
from functools import reduce
from typing import Optional
from .profiler import Profiler
import operator


def _repeat_n(operation: AbelianOperation, expression: Expression, size: Expression) -> Optional[Expression]:
    if operation.value == operator.add:
        return expression * size
    if operation.value == operator.mul:
        return operator.pow(expression, size)
    return None  # TODO: expand this


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
            # print(f"context: {SymPyExpression.convert(context).sympy_object}")
            with self.profiler.profile_section("from-constraint"):
                conditional_intervals = from_constraint(index, constraint, context, is_integral, self.profiler)
            return map_leaves_of_if_then_else(conditional_intervals, eliminate_at_leaves)
        except Exception as exc:
            raise AttributeError("disable this for now") from exc
            return BasicQuantifierExpression(operation, index, constraint, body, is_integral)

    def _eliminate_interval(self, operation: AbelianOperation, index: Variable, interval: ClosedInterval, body: Expression,
                            is_integral: bool, context: Context) \
            -> Expression:
        # try to solve symbolically
        result = self._symbolically_eliminate(operation, index, interval, body, is_integral, context)
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

    def _symbolically_eliminate(self, operation: AbelianOperation, index: Variable, interval: ClosedInterval,
                                body: Expression,
                                is_integral: bool, context: Context) \
            -> Optional[Expression]:
        if context.is_known_to_imply(interval.upper_bound <= interval.lower_bound):
            return BasicConstant(0)

        if operation.value == operator.add:
            if not is_integral:
                if (result := self.symbolic_sum(body, index, interval)) is not None:
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

    def symbolic_sum(self, body: Expression, index: Variable, interval: ClosedInterval) -> Optional[Expression]:
        return SymPyExpression.symbolic_sum(body, index, interval.lower_bound, interval.upper_bound)

    def symbolic_integral(self, body: Expression, index: Variable, interval: ClosedInterval) -> Optional[Expression]:
        if not isinstance(interval.lower_bound, Expression):
            raise NotImplementedError(type(interval.lower_bound))
        if not isinstance(interval.upper_bound, Expression):
            raise NotImplementedError(type(interval.upper_bound))
        # print(f"{Eliminator.integration_counter}th integration: \n"
        #       f"({SymPyExpression.convert(interval.lower_bound).sympy_object},\n"
        #       f"{SymPyExpression.convert(interval.upper_bound).sympy_object})\n {SymPyExpression.convert(body).sympy_object}")
        with self.profiler.profile_section("integration"):
            if DRY_RUN:
                return BasicConstant(0.0)
            else:
                result = SymPyExpression.symbolic_integral(body, index, interval.lower_bound, interval.upper_bound, self.profiler)
                # result = SymPyExpression.symbolic_integral_cached(body, index, interval.lower_bound, interval.upper_bound, self.profiler)
                # print(f"done. {result.sympy_object}")
                return result


DRY_RUN = False
