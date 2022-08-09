from typing import Callable
from .expression import Expression, FunctionApplication, Constant
import neuralpp.symbolic.functions as functions
from .constants import if_then_else


def map_leaves_of_if_then_else(conditional_intervals: Expression,
                               function: Callable[[Expression], Expression]) -> Expression:
    """
    @param conditional_intervals: an if-then-else tree
    @param function: map function
    @return: an Expression with each DottedIntervals i mapped to f(i)
    """
    match conditional_intervals:
        case FunctionApplication(function=Constant(value=functions.conditional), arguments=[if_, then, else_]):
            return if_then_else(if_,
                                map_leaves_of_if_then_else(then, function),
                                map_leaves_of_if_then_else(else_, function))
        case _:
            return function(conditional_intervals)
