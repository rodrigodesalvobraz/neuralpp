import builtins
import fractions
import operator
import typing
from typing import Callable, Dict, List, Optional, Type

import sympy
from sympy import abc
import z3

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


def get_type_from_z3_object(z3_object: z3.ExprRef | z3.FuncDeclRef) -> ExpressionType:
    """
    Z3 uses the word 'sort' in a similar sense to 'type': e.g., z3.IntSort() and z3.BoolSort().
    z3.ArraySort(z3.IntSort(), z3.IntSort()) is the type of arrays who's indexed by int and whose elements are int.
    However, 'int -> int' is not a sort.
    """
    match z3_object:
        case z3.FuncDeclRef():  # for uninterpreted functions.
            return Callable[
                [
                    z3_sort_to_type(z3_object.domain(i))
                    for i in range(z3_object.arity())
                ],
                z3_sort_to_type(z3_object.range()),
            ]
        case z3.ExprRef():
            # this can be an atom such as "x:int" or "2", or this can also be "1+2".
            # Either way, the type is the return type, so just sort().
            return z3_sort_to_type(z3_object.sort())


sort_type_relation = [
    (z3.IntSort(), int),
    (z3.BoolSort(), bool),
    (z3.RealSort(), fractions.Fraction),
    # (z3.FPSort(11, 53), float)  # FPSort(11,53) is double sort (IEEE754, ebits=11, sbits=53)
    # please refer to test/quick_tests/symbolic/z3_usage_test.py:test_z3_fp_sort() for why z3 floating point is not yet
    # supported
]
sort_type_dict = {sort: type_ for sort, type_ in sort_type_relation}
type_sort_dict = {type_: sort for sort, type_ in sort_type_relation}


def z3_sort_to_type(sort: z3.Sort) -> Type:
    if sort in sort_type_dict:
        return sort_type_dict[sort]
    else:
        raise TypeError(f"Unrecognized z3 sort {sort}")


def type_to_z3_sort(type_: Type) -> z3.Sort:
    if type_ in type_sort_dict:
        return type_sort_dict[type_]
    else:
        raise TypeError(f"Unrecognized type {type_}")


def python_callable_to_z3_function(
    python_callable: Callable, type_: Optional[ExpressionType] = None
) -> z3.FuncDeclRef:
    """
    Note: type_ is *not* the function type of the python_callable, but just the type of argument(s).
    This is not ambiguous because z3 does not allow the arguments (of the listed functions here) to be different,
    e.g., "add: float -> int -> int" is not accepted in z3, nor is "less_than: rational -> int -> bool".
    So the arguments must be of the same type.
    E.g., if python_callable is operator.add whose function type is "int -> int -> int", then type_ should be "int".
    """
    match python_callable:
        # boolean operation
        case operator.and_:
            # We do a bit hack here because z3py does not provide direct access to e.g., "add function"
            # So we have to create a function application in z3 and retrieve its declaration using decl().
            return z3.And(True, True).decl()
        case operator.or_:
            return z3.Or(True, True).decl()
        case operator.invert:
            return z3.Not(True).decl()
        case operator.xor:
            return z3.Xor(True, True).decl()

    x, y = z3.Consts(
        "x y", type_to_z3_sort(type_) if type_ is not None else z3.IntSort()
    )
    match python_callable:
        # comparison and arithmetic are overloaded by z3.
        case operator.le:
            return (x <= y).decl()
        case operator.lt:
            return (x < y).decl()
        case operator.ge:
            return (x >= y).decl()
        case operator.gt:
            return (x > y).decl()
        case operator.eq:
            return (x == y).decl()
        case operator.add:
            return (x + y).decl()
        case operator.sub:
            return (x - y).decl()
        case operator.neg:
            return (-x).decl()
        case operator.mul:
            return (x * y).decl()
        case operator.pow:
            return (x**y).decl()
        # min/max
        case builtins.min:
            raise NotImplementedError(
                "Cannot convert min to a z3 function declaration."
                "However we can create z3.If(x<y, x, y) for min(x,y)."
            )
        case builtins.max:
            # if len(arguments) != 2:
            #     raise NotImplementedError("Only 2-element max is supported")
            # return z3.If(arguments[0] > arguments[1], arguments[0], arguments[1])
            raise NotImplementedError(
                "Cannot convert min to a z3 function declaration."
                "However we can create z3.If(x>y, x, y) for max(x,y)."
            )
        # if then else
        case functions.conditional:
            return z3.If(x > y, x, y).decl()  # "x>y" is just a placeholder boolean.
        case _:
            raise ValueError(f"Python callable {python_callable} is not recognized.")


def z3_function_to_python_callable(z3_function: z3.FuncDeclRef) -> Callable:
    match z3_function.kind():
        # boolean operation
        case z3.Z3_OP_AND:
            return operator.and_
        case z3.Z3_OP_OR:
            return operator.or_
        case z3.Z3_OP_NOT:
            return operator.invert
        case z3.Z3_OP_XOR:
            return operator.xor
        # comparison
        case z3.Z3_OP_LE:
            return operator.le
        case z3.Z3_OP_LT:
            return operator.lt
        case z3.Z3_OP_GE:
            return operator.ge
        case z3.Z3_OP_GT:
            return operator.gt
        case z3.Z3_OP_EQ:
            return operator.eq
        case z3.Z3_OP_DISTINCT:
            return operator.ne
        # arithmetic
        case z3.Z3_OP_ADD:
            return operator.add
        case z3.Z3_OP_SUB:
            return operator.sub
        case z3.Z3_OP_UMINUS:
            return operator.neg
        case z3.Z3_OP_MUL:
            return operator.mul
        case z3.Z3_OP_POWER:
            return operator.pow
        # if then else
        case z3.Z3_OP_ITE:
            return functions.conditional
        case _:
            raise ValueError(f"Z3 function {z3_function} is not recognized.")


# On the type of *arguments:
# https://peps.python.org/pep-0484/#arbitrary-argument-lists-and-default-argument-values
def apply_python_callable_on_z3_arguments(
    python_callable: Callable, *arguments: z3.BoolRef | z3.ArithRef
) -> z3.ExprRef:
    """
    Directly calling this function can do something that _python_callable_to_z3_function(python_callable)(arguments)
    cannot do:
        `_python_callable_to_z3_function(builtins.min)(x, y)` raises an error, because "min" cannot be turned
        into a z3 function.
    while
        `_apply_python_callable_on_z3_arguments(builtins.min, x, y)` is fine, because "min(x,y)" can be turned
        into a z3 function application (namely, If(x<y, x, y))
    """
    match python_callable:
        # boolean operation
        case operator.and_:
            return z3.And(arguments)
        case operator.or_:
            return z3.Or(arguments)
        case operator.invert:
            return z3.Not(arguments[0])
        case operator.xor:
            return z3.Xor(arguments[0], arguments[1])
        # comparison
        case operator.le:
            return arguments[0] <= arguments[1]
        case operator.lt:
            return arguments[0] < arguments[1]
        case operator.ge:
            return arguments[0] >= arguments[1]
        case operator.gt:
            return arguments[0] > arguments[1]
        case operator.eq:
            return arguments[0] == arguments[1]
        case operator.ne:
            return arguments[0] != arguments[1]
        # arithmetic
        case operator.add:
            return arguments[0] + arguments[1]
        case operator.sub:
            return arguments[0] - arguments[1]
        case operator.neg:
            return -arguments[0]
        case operator.mul:
            return arguments[0] * arguments[1]
        case operator.pow:
            return arguments[0] ** arguments[1]
        # min/max
        case builtins.min:
            return z3.If(arguments[0] < arguments[1], arguments[0], arguments[1])
        case builtins.max:
            return z3.If(arguments[0] > arguments[1], arguments[0], arguments[1])
        # our functions
        case functions.conditional:
            return z3.If(arguments[0], arguments[1], arguments[2])
        case _:
            raise ValueError(f"Python callable {python_callable} is not recognized.")
