from typing import Any, List
from abc import ABC, abstractmethod


class Expression(ABC):
    @abstractmethod
    # 'Expression': forward reference for recursive type reference https://peps.python.org/pep-0484/#forward-references
    def subexpression(self) -> List['Expression']:
        """
        Returns a list of subexpression.
        E.g.,
        subexpression(f(x,y)) = [f,x,y]
        subexpression(add(a,1)) = [add,a,1]
        """
        pass

    @abstractmethod
    def set(self, i: int, new_expression: 'Expression') -> 'Expression':
        """
        Set i-th subexpression to new_expression. Count from 0.
        E.g.,
        f(x,y).set(1,z) = [f,z,y].
        If it's out of the scope, return error.
        """
        pass

    @abstractmethod
    def replace(self, from_expression: 'Expression', to_expression: 'Expression') -> 'Expression':
        """
        Every expression is immutable so replace() returns either self or a new Expression.
        No in-place modification should be made.
        If from_expression is not in the expression, returns self.
        """
        pass

    def contains(self, target: 'Expression') -> bool:
        """
        Checks if `target` is contained in `self`. The check is deep. E.g.,
        f(x,f(a,b)).contains(a) == True
        """
        for sub_expr in self.subexpression():
            if sub_expr == target or sub_expr.contains(target):
                return True
        return False

    @abstractmethod
    def __eq__(self, other):
        pass


class Variable(Expression):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def subexpression(self) -> List[Expression]:
        return []

    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        return self

    def set(self, i: int, new_expression: Expression) -> Expression:
        raise IndexError("Variable has no subexpression, so you cannot set().")

    def __eq__(self, other):
        match other:
            case Variable(name=other_name):
                return self._name == other_name
            case _:
                return False


class Constant(Expression):
    def __init__(self, value: Any):
        self._value = value

    @property
    def value(self) -> Any:
        return self._value

    def subexpression(self) -> List[Expression]:
        return []

    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        return self

    def set(self, i: int, new_expression: Expression) -> Expression:
        raise IndexError("Constant has no subexpression, so you cannot set().")

    def __eq__(self, other):
        match other:
            case Constant(value=other_value):
                return self._value == other_value
            case _:
                return False


class FunctionApplication(Expression):
    __match_args__ = ("function", "arguments")

    def __init__(self, func: Expression, args: List[Expression]):
        """`func` is an expression. Legal options are:
        1. a Python Callable. E.g.,
            BasicFunctionApplication(BasicConstant(lambda x, y: x + y), [..])

        2. a subclass of Function (function.py). E.g.,
            BasicFunctionApplication(BasicConstant(function.Add()), [..])

        3. a Variable. In this case the function is uninterpreted. E.g.,
        BasicFunctionApplication(BasicConstant(BasicVariable("f")), [..])
        """
        self._subexpression = [func] + args

    @property
    def function(self) -> Expression:
        return self._subexpression[0]

    @property
    def arguments(self) -> List[Expression]:
        return self._subexpression[1:]

    def subexpression(self) -> List[Expression]:
        return self._subexpression

    def __eq__(self, other):
        match other:
            case FunctionApplication(function=func, arguments=args):
                return self.subexpression() == [func] + args
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
        new_subexpressions = list(map(
            lambda e: to_expression if e == from_expression else e.replace(from_expression, to_expression),
            self.subexpression()))
        return FunctionApplication(new_subexpressions[0], new_subexpressions[1:])
