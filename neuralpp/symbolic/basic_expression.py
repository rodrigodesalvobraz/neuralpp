from __future__ import annotations
from neuralpp.symbolic.expression import Expression, FunctionApplication, Variable, Constant, AtomicExpression
from abc import ABC
from typing import Any, List


class BasicExpression(Expression, ABC):
    def new_constant(self, value: Any) -> BasicConstant:
        return BasicConstant(value)

    def new_variable(self, name: str) -> BasicVariable:
        return BasicVariable(name)

    def new_function_application(self, func: Expression, args: List[Expression]) -> BasicFunctionApplication:
        return BasicFunctionApplication(func, args)


class BasicAtomicExpression(BasicExpression, AtomicExpression, ABC):
    def __init__(self, atom: Any):
        self._atom = atom

    @property
    def atom(self) -> str:
        return self._atom


class BasicVariable(BasicAtomicExpression, Variable):
    def __init__(self, name: str):
        BasicAtomicExpression.__init__(self, name)


class BasicConstant(BasicAtomicExpression, Constant):
    def __init__(self, value: Any):
        BasicAtomicExpression.__init__(self, value)


class BasicFunctionApplication(BasicExpression, FunctionApplication):
    def __init__(self, function: Expression, arguments: List[Expression]):
        self._subexpressions = [function] + arguments

    @property
    def function(self) -> Expression:
        return self._subexpressions[0]

    @property
    def arguments(self) -> List[Expression]:
        return self._subexpressions[1:]

    def subexpressions(self) -> List[Expression]:
        return self._subexpressions
