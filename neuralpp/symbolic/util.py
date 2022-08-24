import builtins
import fractions
import operator
import typing
from typing import Callable, List, Type


# typing.Callable can be ambiguous.
# Consider the following two tests:
#   >>> isinstance(int, Type)
#   True
#   >>> isinstance(1, Type)
#   False
# So "int is an instance of Type" and "1 is not an instance of Type"
# But for Callable, we have
#   >>> isinstance(Callable[[int],int], Callable)
#   True
#   >>> isinstance(lambda x: x, Callable)
#   True
# Here the `Callable` in each case means differently, which makes `Callable` ambiguous.
#
# Imagine a typed language, we'd have
#   >>> isinstance(lambda (x: int) : int := x, int->int)
#   True
# and only one of the following two cases can be true:
# CASE ONE: `Callable` means "the type of all function types".
#   >>> isinstance(int->int, Callable)
#   True
#   >>> isinstance(lambda (x: int) : int := x, Callable)
#   False
# or
# CASE TWO: `Callable` means "the type of all functions"
#   >>> isinstance(int->int, Callable)
#   False
#   >>> isinstance(lambda (x: int) : int := x, Callable)
#   True
#
# It should be noted that our usage of `Callable` here means the first, i.e., "the type of all function types"
# And in our code we use `isinstance()` for cases like `isinstance(Callable[[int],int], Callable)`.


ExpressionType = Callable | Type


def return_type_after_application(
    callable_: Callable, number_of_arguments: int
) -> ExpressionType:
    """Given number_of_arguments (<=arity), return the return type after (partial) application."""
    argument_types, return_type = typing.get_args(callable_)
    arity = len(argument_types)
    if number_of_arguments > arity:
        raise ValueError(f"number_of_arguments {number_of_arguments} > arity {arity}.")
    elif number_of_arguments == arity:
        return return_type
    else:
        return Callable[argument_types[number_of_arguments:], return_type]


type_order_in_arithmetic = [fractions.Fraction, float, int]


def get_arithmetic_function_type_from_argument_types(
    argument_types: List[ExpressionType],
) -> Callable:
    try:
        # E.g., if float + int, the return type is float
        return_type = type_order_in_arithmetic[
            min(map(type_order_in_arithmetic.index, argument_types))
        ]
        return Callable[argument_types, return_type]
    except ValueError as err:
        raise ValueError(
            f"Can only infer the return type from arithmetic argument types: "
            f"fractions.Fraction, float and int. {err}"
        )


def boolean_function_of_arity(arity: int) -> Callable:
    return Callable[[bool for i in range(arity)], bool]


def infer_python_callable_type(
    python_callable: Callable, argument_types: List[ExpressionType] | None = None
) -> Callable:
    match python_callable:
        # boolean operation
        case operator.and_ | operator.or_ | operator.xor:
            if argument_types is None:
                # this is also ambiguous because the arity could be arbitrary
                raise AmbiguousTypeError(python_callable)
            if not all([argument_type == bool for argument_type in argument_types]):
                raise TypeError(
                    f"Argument types to boolean function {python_callable} should all be booleans. "
                    f"Got {argument_types}."
                )
            return Callable[argument_types, bool]
        case operator.invert:
            if argument_types is not None and argument_types != [bool]:
                raise TypeError(
                    f"Invert expect only one boolean argument. Got {argument_types}."
                )
            return Callable[[bool], bool]
        # comparison
        case operator.le | operator.lt | operator.ge | operator.gt | operator.eq | operator.ne:
            if argument_types is None:
                raise AmbiguousTypeError(python_callable)
            return Callable[argument_types, bool]
        # arithmetic and min/max
        case operator.add | operator.mul | operator.pow | operator.sub | builtins.min | builtins.max:
            if argument_types is None:
                raise AmbiguousTypeError(python_callable)
            return get_arithmetic_function_type_from_argument_types(argument_types)
        case operator.neg:
            if argument_types is None:
                raise AmbiguousTypeError(python_callable)
            if len(argument_types) != 1:
                raise TypeError(f"Neg only expects one argument.")
            return Callable[argument_types, argument_types[0]]
        # if then else
        case functions.conditional:
            if argument_types is None:
                raise AmbiguousTypeError(python_callable)
            if (
                len(argument_types) != 3
                or argument_types[0] != bool
                or argument_types[1] != argument_types[2]
            ):
                raise TypeError("Wrong conditional expression type.")
            return Callable[argument_types, argument_types[1]]
        case _:
            raise ValueError(f"Python callable {python_callable} is not recognized.")
