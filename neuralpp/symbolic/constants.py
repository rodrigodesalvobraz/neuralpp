import fractions
import operator
from typing import Callable, Type
from .basic_expression import BasicConstant
from .functions import conditional


def if_then_else(type_: Type):
    return BasicConstant(conditional, Callable[[bool, type_, type_], type_])


bool_if_then_else = if_then_else(bool)
int_if_then_else = if_then_else(int)
float_if_then_else = if_then_else(float)
real_if_then_else = if_then_else(fractions.Fraction)


def add(type_: Type):
    return BasicConstant(operator.add, Callable[[type_, type_], type_])


int_add = add(int)
float_add = add(float)
real_add = add(fractions.Fraction)


def multiply(type_: Type):
    return BasicConstant(operator.mul, Callable[[type_, type_], type_])


int_multiply = multiply(int)
float_multiply = multiply(float)
real_multiply = multiply(fractions.Fraction)


def minus(type_: Type):
    return BasicConstant(operator.sub, Callable[[type_, type_], type_])


int_minus = minus(int)
float_minus = minus(float)
real_minus = minus(fractions.Fraction)


def div(type_: Type):
    return BasicConstant(operator.truediv, Callable[[type_, type_], type_])


int_div = div(int)
float_div = div(float)
real_div = div(fractions.Fraction)


def lt(type_: Type):
    return BasicConstant(operator.lt, Callable[[type_, type_], bool])


int_lt = lt(int)
float_lt = lt(float)
real_lt = lt(fractions.Fraction)


def le(type_: Type):
    return BasicConstant(operator.le, Callable[[type_, type_], bool])


int_le = le(int)
float_le = le(float)
real_le = le(fractions.Fraction)


def gt(type_: Type):
    return BasicConstant(operator.gt, Callable[[type_, type_], bool])


int_gt = gt(int)
float_gt = gt(float)
real_gt = gt(fractions.Fraction)


def ge(type_: Type):
    return BasicConstant(operator.ge, Callable[[type_, type_], bool])


int_ge = ge(int)
float_ge = ge(float)
real_ge = ge(fractions.Fraction)


def eq(type_: Type):
    return BasicConstant(operator.eq, Callable[[type_, type_], bool])


int_eq = eq(int)
float_eq = eq(float)
real_eq = eq(fractions.Fraction)


def ne(type_: Type):
    return BasicConstant(operator.ne, Callable[[type_, type_], bool])


int_ne = ne(int)
float_ne = ne(float)
real_ne = ne(fractions.Fraction)
