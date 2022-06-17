from __future__ import annotations  # to support forward reference for recursive type reference

import typing
from typing import List, Any, Optional, Type, Callable
from abc import ABC, abstractmethod


ExpressionType = Callable | Type


def return_type_after_application(callable_: Callable, number_of_arguments: int) -> ExpressionType:
    """ Given number_of_arguments (<=arity), return the return type after (partial) application. """
    argument_types, return_type = typing.get_args(callable_)
    arity = len(argument_types)
    if number_of_arguments > arity:
        raise ValueError(f"number_of_arguments {number_of_arguments} > arity {arity}.")
    elif number_of_arguments == arity:
        return return_type
    else:
        return Callable[argument_types[number_of_arguments:], return_type]


class Expression(ABC):
    def __init__(self, expression_type: ExpressionType):
        self._type = expression_type

    @property
    @abstractmethod
    def subexpressions(self) -> List[Expression]:
        """
        Returns a list of subexpressions.
        E.g.,
        subexpressions(f(x,y)) = [f,x,y]
        subexpressions(add(a,1)) = [add,a,1]
        """
        pass

    @abstractmethod
    def set(self, i: int, new_expression: Expression) -> Expression:
        """
        Set i-th subexpressions to new_expression. Count from 0.
        E.g.,
        f(x,y).set(1,z) = [f,z,y].
        If it's out of the scope, return error.
        """
        pass

    @abstractmethod
    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        """
        Every expression is immutable so replace() returns either self or a new Expression.
        No in-place modification should be made.
        If from_expression is not in the expression, returns self.
        """
        pass

    def contains(self, target: Expression) -> bool:
        """
        Checks if `target` is contained in `self`. The check is deep. An expression contains itself. E.g.,
        f(x,f(a,b)).contains(a) == True
        a.contains(a) == True
        """
        if self == target:
            return True

        for sub_expr in self.subexpressions:
            if sub_expr.contains(target):
                return True

        return False

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @classmethod
    @abstractmethod
    def new_constant(cls, value: Any, type_: Optional[ExpressionType]) -> Constant:
        pass

    @classmethod
    @abstractmethod
    def new_variable(cls, name: str, type_: ExpressionType) -> Variable:
        pass

    @classmethod
    @abstractmethod
    def new_function_application(cls, function: Expression, arguments: List[Expression]) -> FunctionApplication:
        pass

    @classmethod
    def _convert(cls, from_expression: Expression) -> Expression:
        """ general helper for converting an Expression into this subclass of Expression. """
        if isinstance(from_expression, cls):
            return from_expression
        match from_expression:
            case Constant(value=value, type=type_):
                return cls.new_constant(value, type_)
            case Variable(name=name, type=type_):
                return cls.new_variable(name, type_)
            case FunctionApplication(function=function, arguments=arguments):
                return cls.new_function_application(function, arguments)
            case _:
                raise ValueError(f"invalid from_expression {from_expression}: {type(from_expression)}")

    @property
    def type(self) -> ExpressionType:
        return self._type

    def get_function_type(self) -> ExpressionType:
        """
        For example, a function (the declaration, not the application) can be:
            add_func_type = Callable[[int, int], int]
            add_func = Constant(operator.add, add_func_type)
        get_function_type() serves as a method to retrieve the "function type" of `add_func`, so we can expect:
            add_func.get_function_type() == add_func_type
        Note add_func can also be
            add_func = Variable("add", add_func_type)  # uninterpreted function
        or
            three_way_add = Variable("three_way_add", Callable[[int, int, int], int])
            add_func = FunctionApplication(three_way_add, Constant(0))
        In both cases we can expect
            add_func.get_function_type() == add_func_type
        """
        match self:
            case Constant(type=type_) | Variable(type=type_):
                return type_
            case FunctionApplication(function=function, number_of_arguments=num):
                return function.get_return_type(num)

    def get_return_type(self, number_of_arguments: int) -> ExpressionType:
        function_type = self.get_function_type()
        if not isinstance(function_type, Callable):
            raise TypeError(f"{self}'s function is not of function type.")
        return return_type_after_application(function_type, number_of_arguments)


class AtomicExpression(Expression, ABC):
    @property
    @abstractmethod
    def base_type(self) -> str:
        pass

    @property
    @abstractmethod
    def atom(self) -> str:
        pass

    @property
    def subexpressions(self) -> List[Expression]:
        return []

    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        if from_expression == self:
            return to_expression
        else:
            return self

    def set(self, i: int, new_expression: Expression) -> Expression:
        raise IndexError(f"{type(self)} has no subexpressions, so you cannot set().")

    def __eq__(self, other) -> bool:
        match other:
            case AtomicExpression(base_type=other_base_type, atom=other_atom, type=other_type):
                return other_base_type == self.base_type and self.atom == other_atom and self.type == other_type
            case _:
                return False

    def contains(self, target: Expression) -> bool:
        return self == target


class Variable(AtomicExpression, ABC):
    @property
    def base_type(self) -> str:
        return "Variable"

    @property
    def name(self) -> str:
        return self.atom


class Constant(AtomicExpression, ABC):
    @property
    def base_type(self) -> str:
        return "Constant"

    @property
    def value(self) -> Any:
        return self.atom


class FunctionApplication(Expression, ABC):
    __match_args__ = ("function", "arguments", "number_of_arguments")

    @property
    @abstractmethod
    def function(self) -> Expression:
        pass

    @property
    @abstractmethod
    def number_of_arguments(self) -> int:
        # in some implementation getting number_of_arguments without calling arguments is useful,
        # e.g., a lazy implementation where arguments are only evaluated when used
        # Note: this is not `arity`, which is a property of a function, not a function application.
        pass

    @property
    @abstractmethod
    def arguments(self) -> List[Expression]:
        pass

    @property
    @abstractmethod
    def subexpressions(self) -> List[Expression]:
        pass

    def __eq__(self, other):
        match other:
            case FunctionApplication(function=function, arguments=arguments, type=other_type):
                return self.subexpressions == [function] + arguments and self.type == other_type
            case _:
                return False

    def set(self, i: int, new_expression: Expression) -> Expression:
        if i == 0:
            return self.new_function_application(new_expression, self.arguments)

        # evaluate len after i != 0, if i == 0 we can be lazy
        if i-1 < self.number_of_arguments:
            arguments = self.arguments
            arguments[i-1] = new_expression
            return self.new_function_application(self.function, arguments)
        else:
            raise IndexError(f"Out of scope. Function application only has {self.number_of_arguments} arguments "
                             f"but you are setting {i-1}th arguments.")

    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        # recursively do the replacement
        new_subexpressions = [
            to_expression if e == from_expression else e.replace(from_expression, to_expression)
            for e in self.subexpressions
        ]
        return self.new_function_application(new_subexpressions[0], new_subexpressions[1:])


class NotTypedError(ValueError, TypeError):
    pass


class VariableNotTypedError(NotTypedError):
    pass


class FunctionNotTypedError(NotTypedError):
    pass
