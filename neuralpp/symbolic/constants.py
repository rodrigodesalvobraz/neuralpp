import fractions
import operator
from typing import Callable, Type, Any
from .basic_expression import BasicConstant
from .functions import conditional
from .expression import Expression


def if_then_else_function(type_: Type) -> Expression:
    return BasicConstant(conditional, Callable[[bool, type_, type_], type_])


bool_if_then_else_function = if_then_else_function(bool)
int_if_then_else_function = if_then_else_function(int)
float_if_then_else_function = if_then_else_function(float)
real_if_then_else_function = if_then_else_function(fractions.Fraction)


def if_then_else(if_: Any, then_: Any, else_: Any) -> Expression:
    if not type(then_) == type(else_):
        raise TypeError(f"Expect then-clause ({type(then_)}) and else-clause ({type(else_)}) have the same type.")
    match then_:
        case bool():
            return bool_if_then_else_function(if_, then_, else_)
        case int():
            return int_if_then_else_function(if_, then_, else_)
        case float():
            return float_if_then_else_function(if_, then_, else_)
        case fractions.Fraction():
            return real_if_then_else_function(if_, then_, else_)
        case _:
            raise TypeError(f"Unrecognized type {type(then_)}.")


def add(type_: Type) -> Expression:
    return BasicConstant(operator.add, Callable[[type_, type_], type_])


int_add = add(int)
float_add = add(float)
real_add = add(fractions.Fraction)


def multiply(type_: Type) -> Expression:
    return BasicConstant(operator.mul, Callable[[type_, type_], type_])


int_multiply = multiply(int)
float_multiply = multiply(float)
real_multiply = multiply(fractions.Fraction)


def minus(type_: Type) -> Expression:
    return BasicConstant(operator.sub, Callable[[type_, type_], type_])


int_minus = minus(int)
float_minus = minus(float)
real_minus = minus(fractions.Fraction)


def div(type_: Type) -> Expression:
    return BasicConstant(operator.truediv, Callable[[type_, type_], type_])


int_div = div(int)
float_div = div(float)
real_div = div(fractions.Fraction)


def lt(type_: Type) -> Expression:
    return BasicConstant(operator.lt, Callable[[type_, type_], bool])


int_lt = lt(int)
float_lt = lt(float)
real_lt = lt(fractions.Fraction)


def le(type_: Type) -> Expression:
    return BasicConstant(operator.le, Callable[[type_, type_], bool])


int_le = le(int)
float_le = le(float)
real_le = le(fractions.Fraction)


def gt(type_: Type) -> Expression:
    return BasicConstant(operator.gt, Callable[[type_, type_], bool])


int_gt = gt(int)
float_gt = gt(float)
real_gt = gt(fractions.Fraction)


def ge(type_: Type) -> Expression:
    return BasicConstant(operator.ge, Callable[[type_, type_], bool])


int_ge = ge(int)
float_ge = ge(float)
real_ge = ge(fractions.Fraction)


def eq(type_: Type) -> Expression:
    return BasicConstant(operator.eq, Callable[[type_, type_], bool])


int_eq = eq(int)
float_eq = eq(float)
real_eq = eq(fractions.Fraction)


def ne(type_: Type) -> Expression:
    return BasicConstant(operator.ne, Callable[[type_, type_], bool])


int_ne = ne(int)
float_ne = ne(float)
real_ne = ne(fractions.Fraction)
