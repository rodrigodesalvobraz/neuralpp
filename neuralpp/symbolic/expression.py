from __future__ import annotations  # to support forward reference for recursive type reference
from typing import Any, List, Type
from abc import ABC, abstractmethod


class Expression(ABC):
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
    def new_constant(cls, value: Any) -> Constant:
        pass

    @classmethod
    @abstractmethod
    def new_variable(cls, name: str) -> Variable:
        pass

    @classmethod
    @abstractmethod
    def new_function_application(cls, func: Expression, args: List[Expression]) -> FunctionApplication:
        pass


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
            case AtomicExpression(base_type=other_base_type, atom=other_atom):
                return other_base_type == self.base_type and self.atom == other_atom
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
    __match_args__ = ("function", "arguments")

    @property
    @abstractmethod
    def function(self) -> Expression:
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
            case FunctionApplication(function=function, arguments=arguments):
                return self.subexpressions == [function] + arguments
            case _:
                return False

    def set(self, i: int, new_expression: Expression) -> Expression:
        if i == 0:
            return self.new_function_application(new_expression, self.arguments)

        arity = len(self.arguments)  # evaluate len after i != 0, if i == 0 we can be lazy
        if i-1 < arity:
            arguments = self.arguments
            arguments[i-1] = new_expression
            return self.new_function_application(self.function, arguments)
        else:
            raise IndexError(f"Out of scope. Function only has arity {arity} but you are setting {i-1}th arguments.")

    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        # recursively do the replacement
        new_subexpressions = [
            to_expression if e == from_expression else e.replace(from_expression, to_expression)
            for e in self.subexpressions
        ]
        return self.new_function_application(new_subexpressions[0], new_subexpressions[1:])
