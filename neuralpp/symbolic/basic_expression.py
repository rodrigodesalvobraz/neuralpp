from __future__ import annotations

import operator
import builtins
from neuralpp.symbolic.expression import Expression, FunctionApplication, Variable, Constant, AtomicExpression, \
    VariableNotTypedError, ExpressionType
from abc import ABC
from typing import Any, List, Optional, Type, Callable, Tuple


class AmbiguousTypeError(TypeError, ValueError):
    def __init__(self, python_callable: Callable):
        super().__init__(f"{python_callable} is ambiguous.")


def infer_python_callable_type(python_callable: Callable) -> ExpressionType:
    match python_callable:
        # boolean operation
        case operator.and_ | operator.or_ | operator.xor:
            raise AmbiguousTypeError(python_callable)  # this is also ambiguous because the arity could be arbitrary
            # return Callable[[bool, bool], bool]
        case operator.not_:
            return Callable[[bool], bool]
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
    def new_constant(cls, value: Any, type_: Optional[ExpressionType] = None) -> BasicConstant:
        return BasicConstant(value, type_)

    @classmethod
    def new_variable(cls, name: str, type_: ExpressionType) -> BasicVariable:
        return BasicVariable(name, type_)

    @classmethod
    def new_function_application(cls, function: Expression, arguments: List[Expression]) -> BasicFunctionApplication:
        return BasicFunctionApplication(function, arguments)

    @classmethod
    def pythonize_value(cls, value: Any) -> Any:
        return value


class BasicAtomicExpression(BasicExpression, AtomicExpression, ABC):
    def __init__(self, atom: Any, expression_type: Optional[Expression] = None):
        if expression_type is None and not isinstance(atom, ExpressionType):
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
    def __init__(self, name: str, type_: Expression):
        if type_ is None:
            raise VariableNotTypedError
        BasicAtomicExpression.__init__(self, name, type_)


class BasicConstant(BasicAtomicExpression, Constant):
    def __init__(self, value: Any, type_: Optional[Expression] = None):
        BasicAtomicExpression.__init__(self, value, type_)

    @staticmethod
    def atom_compare(atom1: Any, atom2: Any) -> bool:
        return atom1 == atom2


class BasicFunctionApplication(BasicExpression, FunctionApplication):
    def __init__(self, function: Expression, arguments: List[Expression]):
        function_type = function.get_return_type(len(arguments))
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
