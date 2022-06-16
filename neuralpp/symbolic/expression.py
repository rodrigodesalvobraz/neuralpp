from __future__ import annotations  # to support forward reference for recursive type reference

from typing import List, Any, Optional, Tuple, Type, NewType
from abc import ABC, abstractmethod


class FunctionType:
    def __init__(self, argument_types: List[Type], return_type: Type):
        self._argument_types = argument_types
        self._return_type = return_type

    @property
    def argument_types(self):
        return self._argument_types

    @property
    def return_type(self):
        return self._return_type

    @property
    def arity(self):
        return len(self._argument_types)

    def return_type_after_application(self, number_of_arguments: int) -> FunctionType | Type:
        """ Given number_of_arguments (<=arity), return the return type after (partial) application. """
        if number_of_arguments > self.arity:
            raise ValueError(f"number_of_arguments {number_of_arguments} > arity {self.arity}.")
        elif number_of_arguments == self.arity:
            return self.return_type
        else:
            return FunctionType(self.argument_types[number_of_arguments:], self.return_type)

    def __str__(self) -> str:
        return f"{self.argument_types} -> {self.return_type}"


class Expression(ABC):
    def __init__(self, expression_type: Expression):
        """
        All types are represented by a function-style 2-element tuple (argument_types, return_type),
        where argument_types is a list of argument types.
        For simple type such as int, it's treated as an arity=0 function, i.e., ([], int).
        We refer to this tuple type representation as "internal type".

        The reason I do not use "typing.Callable | int | float ..." is that it's not a unified system,
        for example, we have
            not isinstance(typing.Callable, type) and isinstance(int, type)
            not isinstance(typing.Callable, typing.Type) and isinstance(int, typing.Type)
        """
        self._type = expression_type

    @staticmethod
    def is_internal_type(atom: Any):
        return isinstance(atom, FunctionType | Type)

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
    def new_constant(cls, value: Any, constant_type: Optional[Expression]) -> Constant:
        pass

    @classmethod
    @abstractmethod
    def new_variable(cls, name: str, variable_type: Expression) -> Variable:
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
            case Constant(value=value, type=constant_type):
                return cls.new_constant(value, constant_type)
            case Variable(name=name, type=variable_type):
                return cls.new_variable(name, variable_type)
            case FunctionApplication(function=function, arguments=arguments, type=function_type):
                return cls.new_function_application(function, arguments)
            case _:
                raise ValueError(f"invalid from_expression {from_expression}: {type(from_expression)}")

    @property
    def type(self) -> Expression:
        return self._type

    def get_function_type(self) -> FunctionType | Type:
        """ Call me when self is a function """
        match self:
            case Constant(type=type_expression) | Variable(type=type_expression):
                return type_expression.get_function_type_of_type_expression()
            case FunctionApplication(function=function, number_of_arguments=num):
                return function.get_return_type(num)

    def get_function_type_of_type_expression(self) -> FunctionType | Type:
        """ Call me when self is a type Expression of a function """
        match self:
            case Constant(value=internal_type):
                if not Expression.is_internal_type(internal_type):
                    raise TypeError(f"{internal_type} is not internal type.")
                return internal_type
            case _:
                raise TypeError(f"Unrecognized type expression {self}.")

    def get_return_type(self, number_of_arguments: int) -> FunctionType | Type:
        function_type = self.get_function_type()
        return function_type.return_type_after_application(number_of_arguments)


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
