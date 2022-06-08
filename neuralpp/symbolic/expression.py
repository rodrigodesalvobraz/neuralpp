from __future__ import annotations  # to support forward reference for recursive type reference
from typing import Any, List
from abc import ABC, abstractmethod


class Expression(ABC):
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

        for sub_expr in self.subexpressions():
            if sub_expr.contains(target):
                return True

        return False

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass


class AtomicExpression(Expression):
    def __init__(self, atom: Any):
        self._atom = atom

    @property
    @abstractmethod
    def base_type(self) -> str:
        pass

    @property
    def atom(self) -> str:
        return self._atom

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
                return other_base_type == self.base_type and self.atom == other.atom
            case _:
                return False

    def contains(self, target: Expression) -> bool:
        return self == target


class Variable(AtomicExpression):
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def base_type(self) -> str:
        return "Variable"

    @property
    def name(self) -> str:
        return super().atom


class Constant(AtomicExpression):
    def __init__(self, value: Any):
        super().__init__(value)

    @property
    def base_type(self) -> str:
        return "Constant"

    @property
    def value(self) -> Any:
        return super().atom


class FunctionApplication(Expression):
    __match_args__ = ("function", "arguments")

    def __init__(self, function: Expression, arguments: List[Expression]):
        """`func` is an expression. Legal options are:
        1. a Python Callable. E.g.,
            BasicFunctionApplication(BasicConstant(lambda x, y: x + y), [..])

        2. a subclass of Function (function.py). E.g.,
            BasicFunctionApplication(BasicConstant(function.Add()), [..])

        3. a Variable. In this case the function is uninterpreted. E.g.,
        BasicFunctionApplication(BasicConstant(BasicVariable("f")), [..])
        """
        self._subexpressions = [function] + arguments

    @property
    def function(self) -> Expression:
        return self._subexpressions[0]

    @property
    def arguments(self) -> List[Expression]:
        return self._subexpressions[1:]

    def subexpressions(self) -> List[Expression]:
        return self._subexpressions

    def __eq__(self, other):
        match other:
            case FunctionApplication(function=function, arguments=arguments):
                return self.subexpressions() == [function] + arguments
            case _:
                return False

    def set(self, i: int, new_expression: Expression) -> Expression:
        arity = len(self.arguments)
        if i == 0:
            return FunctionApplication(new_expression, self.arguments)
        elif i-1 < arity:
            arguments = self.arguments
            arguments[i-1] = new_expression
            return FunctionApplication(self.function, arguments)
        else:
            raise IndexError(f"Out of scope. Function only has arity {arity} but you are setting {i-1}th arguments.")

    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        # recursively do the replacement
        new_subexpressions = [
            to_expression if e == from_expression else e.replace(from_expression, to_expression)
            for e in self.subexpressions()
        ]
        return FunctionApplication(new_subexpressions[0], new_subexpressions[1:])
