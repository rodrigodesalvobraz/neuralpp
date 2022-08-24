import builtins
import fractions
import operator
import sympy
from sympy import abc
import typing
from typing import Callable, Dict, List, Type

import neuralpp.symbolic.functions as functions


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


def infer_sympy_object_type(
    sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]
) -> ExpressionType:
    """
    type_dict can be, for example, {a: int, b: int, c: float, f: int->int}.
    """
    match sympy_object:
        case sympy.Integer():
            return int
        case sympy.Float():
            return float
        case sympy.Rational():
            return fractions.Fraction
        case sympy.logic.boolalg.BooleanAtom():
            return bool
        case sympy.Piecewise():  # piecewise is not mapped in python_callable_and_sympy_function_relation
            return infer_sympy_object_type(sympy_object.args[0][0], type_dict)
        case _:
            # We can look up type_dict for variable like `symbol("x")`.
            # A variable can also be an uninterpreted function `sympy.core.function.UndefinedFunction('f')`
            try:
                return type_dict[sympy_object]
            except KeyError:  # if it's not in type_dict, try figure out ourselves
                # Here, sympy_object must be function applications like 'x+y'
                if (
                    len(sympy_object.args) == 0
                ):  # len(sympy_object.args) could raise (e.g, len(sympy.Add.args))
                    raise TypeError("expect function application")
                _, return_type = typing.get_args(
                    infer_sympy_function_type(sympy_object, type_dict)
                )
                return return_type


def infer_sympy_function_type(
    sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]
) -> Callable:
    """
    Assume sympy_object is a function application, return the function type of the function application.
    Note this is different from infer_sympy_object_type, which returns the type of function application.
    E.g.,
    >>> from sympy.abc import a, b
    >>> _infer_sympy_function_type(a+b, {a:int, b:int})
    Callable[[int, int], int]
    >>> _infer_sympy_object_type(a+b, {a:int, b:int})
    int
    """
    return infer_python_callable_type(
        sympy_function_to_python_callable(sympy_object.func),
        [infer_sympy_object_type(arg, type_dict) for arg in sympy_object.args],
    )


sympy_sub = sympy.Lambda((abc.x, abc.y), abc.x - abc.y)
sympy_neg = sympy.Lambda((abc.x,), -abc.x)  # "lambda x: (-1)*x"
sympy_cond = sympy.Lambda(
    (abc.i, abc.t, abc.e), sympy.Piecewise((abc.t, abc.i), (abc.e, True))
)
sympy_div = sympy.Lambda((abc.x, abc.y), abc.x / abc.y)
# Refer to sympy_simplification_test:test_unevaluate() for this design that uses sympy.Lambda()
python_callable_and_sympy_function_relation = [
    # boolean operation
    (operator.and_, sympy.And),
    (operator.or_, sympy.Or),
    (operator.invert, sympy.Not),
    (operator.xor, sympy.Xor),
    # comparison
    (operator.le, sympy.Le),
    (operator.lt, sympy.Lt),
    (operator.ge, sympy.Ge),
    (operator.gt, sympy.Gt),
    (operator.eq, sympy.Eq),
    (operator.ne, sympy.Ne),
    # arithmetic
    (operator.add, sympy.Add),
    (operator.mul, sympy.Mul),
    (operator.pow, sympy.Pow),
    (operator.sub, sympy_sub),
    (operator.neg, sympy_neg),
    (operator.truediv, sympy_div),
    # min/max
    (builtins.min, sympy.Min),
    (builtins.max, sympy.Max),
    # conditional
    (functions.conditional, sympy_cond),
]


sympy_function_python_callable_dict = {
    sympy_function: python_callable
    for python_callable, sympy_function in python_callable_and_sympy_function_relation
}


python_callable_sympy_function_dict = {
    python_callable: sympy_function
    for python_callable, sympy_function in python_callable_and_sympy_function_relation
}


def sympy_function_to_python_callable(sympy_function: sympy.Basic) -> Callable:
    try:
        return sympy_function_python_callable_dict[sympy_function]
    except KeyError:
        if sympy_function == sympy.Piecewise:
            return functions.conditional
        raise ValueError(f"SymPy function {sympy_function} is not recognized.")


def python_callable_to_sympy_function(python_callable: Callable) -> sympy.Basic:
    try:
        return python_callable_sympy_function_dict[python_callable]
    except KeyError:
        raise ValueError(f"Python callable {python_callable} is not recognized.")
