from __future__ import annotations

import operator
import builtins
from neuralpp.symbolic.expression import Expression, FunctionApplication, Variable, Constant, AtomicExpression, \
    VariableNotTypedError, FunctionType
from abc import ABC
from typing import Any, List, Optional, Type, Callable, Tuple


class AmbiguousTypeError(TypeError, ValueError):
    def __init__(self, python_callable: Callable):
        super().__init__(f"{python_callable} is ambiguous.")


def infer_python_callable_type(python_callable: Callable) -> FunctionType:
    match python_callable:
        # boolean operation
        case operator.__and__ | operator.__or__ | operator.__xor__:
            raise AmbiguousTypeError(python_callable)  # this is also ambiguous because the arity could be arbitrary
            # return FunctionType([bool, bool], bool)
        case operator.__not__:
            return FunctionType([bool], bool)
        # comparison
        case operator.le | operator.lt | operator.ge | operator.gt | operator.eq:
            raise AmbiguousTypeError(python_callable)
        # arithmetic
        case operator.add | operator.mul | operator.pow:
            raise AmbiguousTypeError(python_callable)
        # min/max
        case builtins.min | builtins.max:
            raise AmbiguousTypeError(python_callable)
        case _:
            raise ValueError(f"Python callable {python_callable} is not recognized.")


class BasicExpression(Expression, ABC):
    @classmethod
    def new_constant(cls, value: Any, constant_type: Optional[Expression] = None) -> BasicConstant:
        return BasicConstant(value, constant_type)

    @classmethod
    def new_variable(cls, name: str, variable_type: Expression) -> BasicVariable:
        return BasicVariable(name, variable_type)

    @classmethod
    def new_function_application(cls, function: Expression, arguments: List[Expression]) -> BasicFunctionApplication:
        return BasicFunctionApplication(function, arguments)


def new_type(python_type: Type) -> Expression:
    """ A handy method to create properly wrapped type as an Expression. E.g., new_type(int). """
    return BasicConstant(python_type)


class BasicAtomicExpression(BasicExpression, AtomicExpression, ABC):
    def __init__(self, atom: Any, expression_type: Optional[Expression] = None):
        if expression_type is None and not Expression.is_internal_type(atom):
            # try to infer type for atom
            if isinstance(atom, Callable):
                internal_type = infer_python_callable_type(atom)
            else:
                internal_type = type(atom)
            expression_type = BasicConstant(internal_type, None)
        super().__init__(expression_type)
        self._atom = atom

    @property
    def atom(self) -> str:
        return self._atom


class BasicVariable(BasicAtomicExpression, Variable):
    def __init__(self, name: str, variable_type: Expression):
        if variable_type is None:
            raise VariableNotTypedError
        BasicAtomicExpression.__init__(self, name, variable_type)


class BasicConstant(BasicAtomicExpression, Constant):
    def __init__(self, value: Any, constant_type: Optional[Expression] = None):
        BasicAtomicExpression.__init__(self, value, constant_type)

    # just add this one to simplify debugging
    def __str__(self) -> str:
        return f"{self.value}: {self.type}"


class BasicFunctionApplication(BasicExpression, FunctionApplication):
    def __init__(self, function: Expression, arguments: List[Expression]):
        function_type = BasicConstant(function.get_return_type(len(arguments)))
        super().__init__(function_type)
        self._subexpressions = [function] + arguments

    @property
    def function(self) -> Expression:
        return self._subexpressions[0]

    @property
    def arguments(self) -> List[Expression]:
        return self._subexpressions[1:]

    @property
    def subexpressions(self) -> List[Expression]:
        return self._subexpressions

    @property
    def number_of_arguments(self) -> int:
        return len(self.arguments)
