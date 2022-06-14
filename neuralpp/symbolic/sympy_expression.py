from __future__ import annotations
from typing import List, Any, Type, Callable, Tuple
from abc import ABC

import sympy
import operator
import builtins
from neuralpp.symbolic.expression import Expression, FunctionApplication, Variable, Constant

# In this file's doc, I try to avoid the term `sympy expression` because it could mean both sympy.Expr (or sympy.Basic)
# and SymPyExpression. I usually use "sympy object" to refer to the former and "expression" to refer to the latter.


def python_callable_to_sympy_function(python_callable: Callable) -> Type[sympy.Basic]:
    match python_callable:
        # boolean operation
        case operator.__and__:
            return sympy.And
        case operator.__or__:
            return sympy.Or
        case operator.__not__:
            return sympy.Not
        case operator.__xor__:
            return sympy.Xor
        # comparison
        case operator.le:
            return sympy.Le
        case operator.lt:
            return sympy.Lt
        case operator.ge:
            return sympy.Ge
        case operator.gt:
            return sympy.Gt
        case operator.eq:
            return sympy.Eq
        # arithmetic
        case operator.add:
            return sympy.Add
        case operator.mul:
            return sympy.Mul
        case operator.pow:
            return sympy.Pow
        # min/max
        case builtins.min:
            return sympy.Min
        case builtins.max:
            return sympy.Max
        case _:
            raise ValueError(f"Python callable {python_callable} is not recognized.")


def is_sympy_value(sympy_object: sympy.Basic) -> bool:
    return isinstance(sympy_object, sympy.Number) or \
           isinstance(sympy_object, sympy.logic.boolalg.BooleanAtom)


def sympy_object_to_expression(sympy_object: sympy.Basic) -> SymPyExpression:
    # Here we just try to find a type of expression for sympy object.
    if type(sympy_object) == sympy.Symbol:
        return SymPyVariable(sympy_object)
    elif is_sympy_value(sympy_object):
        return SymPyConstant(sympy_object)
    else:
        return SymPyFunctionApplication(sympy_object)


class SymPyExpression(Expression, ABC):
    def __init__(self, sympy_object: sympy.Basic):
        self._sympy_object = sympy_object

    @classmethod
    def new_constant(cls, value: Any) -> SymPyConstant:
        # if a string contains a whitespace it'll be treated as multiple variables in sympy.symbols
        if isinstance(value, sympy.Basic):
            sympy_object = value
        elif type(value) == bool:
            sympy_object = sympy.S.true if value else sympy.S.false
        elif type(value) == int:
            sympy_object = sympy.Integer(value)
        elif type(value) == float:
            sympy_object = sympy.Float(value)
        elif type(value) == str:
            sympy_object = sympy.Function(value)
        else:
            try:
                sympy_object = python_callable_to_sympy_function(value)
            except Exception:
                raise ValueError(f"SymPyConstant does not support {type(value)}: "
                                 f"unable to turn into a sympy representation internally")
        return SymPyConstant(sympy_object)

    @classmethod
    def new_variable(cls, name: str) -> SymPyVariable:
        # if a string contains a whitespace it'll be treated as multiple variables in sympy.symbols
        if ' ' in name:
            raise ValueError(f"`{name}` should not contain a whitespace!")
        sympy_var = sympy.symbols(name)
        return SymPyVariable(sympy_var)

    @classmethod
    def new_function_application(cls, function: Expression, arguments: List[Expression]) -> SymPyFunctionApplication:
        # we cannot be lazy here because the goal is to create a sympy object, so arguments must be
        # recursively converted to sympy object
        match function:
            # first check if function is of SymPyConstant, where sympy_function is assumed to be a sympy function,
            # and we don't need to convert it.
            case SymPyConstant(value=sympy_function):
                return SymPyFunctionApplication(sympy_function(*[cls._convert(argument).sympy_object
                                                                 for argument in arguments],
                                                               evaluate=False))  # otherwise Add(1,1) will be 2 in sympy
            # if function is not of SymPyConstant but of Constant, then it is assumed to be a python callable
            case Constant(value=python_callable):
                # during the call, ValueError will be implicitly raised if we cannot convert
                sympy_function = python_callable_to_sympy_function(python_callable)
                # mutual recursively call _convert().
                # this wll guarantee type of cls._convert(argument) is SymPyExpression
                return SymPyFunctionApplication(sympy_function(*[cls._convert(argument).sympy_object
                                                                 for argument in arguments],
                                                               evaluate=False))
            case Variable(name=name):
                raise ValueError(f"Cannot create a SymPyExpression from uninterpreted function {name}")
            case FunctionApplication(_, _):
                raise ValueError("The function must be a python callable.")
            case _:
                raise ValueError("Unknown case.")

    @property
    def sympy_object(self):
        return self._sympy_object


class SymPyVariable(SymPyExpression, Variable):
    @property
    def atom(self) -> str:
        return str(self._sympy_object)


class SymPyConstant(SymPyExpression, Constant):
    @property
    def atom(self) -> Any:
        return self._sympy_object


class SymPyFunctionApplication(SymPyExpression, FunctionApplication):
    @property
    def function(self) -> Expression:
        return SymPyConstant(self._sympy_object.func)

    @property
    def arguments(self) -> List[Expression]:
        return [sympy_object_to_expression(argument) for argument in self._sympy_object.args]

    @property
    def native_arguments(self) -> Tuple[sympy.Basic, ...]:
        """ faster than arguments """
        return self._sympy_object.args  # sympy f.args returns a tuple

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.function] + self.arguments
