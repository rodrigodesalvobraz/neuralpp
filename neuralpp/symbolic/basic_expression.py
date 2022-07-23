from __future__ import annotations

import operator
import builtins
from neuralpp.symbolic.expression import Expression, FunctionApplication, Variable, Constant, AtomicExpression, \
    VariableNotTypedError, ExpressionType, get_arithmetic_function_type_from_argument_types, Context, \
    QuantifierExpression, AbelianOperation
from abc import ABC
from typing import Any, List, Optional, Callable, Dict, get_args
import neuralpp.symbolic.functions as functions


class AmbiguousTypeError(TypeError, ValueError):
    def __init__(self, python_callable: Callable):
        super().__init__(f"{python_callable} is ambiguous.")


def boolean_function_of_arity(arity: int) -> Callable:
    return Callable[[bool for i in range(arity)], bool]


def infer_python_callable_type(python_callable: Callable, argument_types: List[ExpressionType] | None = None) -> \
        Callable:
    match python_callable:
        # boolean operation
        case operator.and_ | operator.or_ | operator.xor:
            if argument_types is None:
                raise AmbiguousTypeError(python_callable)  # this is also ambiguous because the arity could be arbitrary
            if not all([argument_type == bool for argument_type in argument_types]):
                raise TypeError(f"Argument types to boolean function {python_callable} should all be booleans. "
                                f"Got {argument_types}.")
            return Callable[argument_types, bool]
        case operator.invert:
            if argument_types is not None and argument_types != [bool]:
                raise TypeError(f"Invert expect only one boolean argument. Got {argument_types}.")
            return Callable[[bool], bool]
        # comparison
        case operator.le | operator.lt | operator.ge | operator.gt | operator.eq:
            if argument_types is None:
                raise AmbiguousTypeError(python_callable)
            return Callable[argument_types, bool]
        # arithmetic and min/max
        case operator.add | operator.mul | operator.pow | operator.sub | builtins.min | builtins.max:
            if argument_types is None:
                raise AmbiguousTypeError(python_callable)
            return get_arithmetic_function_type_from_argument_types(argument_types)
        case operator.neg:
            if argument_types is None:
                raise AmbiguousTypeError(python_callable)
            if len(argument_types) != 1:
                raise TypeError(f"Neg only expects one argument.")
            return Callable[argument_types, argument_types[0]]
        # if then else
        case functions.conditional:
            if argument_types is None:
                raise AmbiguousTypeError(python_callable)
            if len(argument_types) != 3 or argument_types[0] != bool or argument_types[1] != argument_types[2]:
                raise TypeError("Wrong conditional expression type.")
            return Callable[argument_types, argument_types[1]]
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
    def new_quantifier_expression(cls,
                                  operation: AbelianOperation, index: Variable, constrain: Context, body: Expression,
                                  ) -> Expression:
        if constrain.satisfiability_is_known and constrain.unsatisfiable:
            return operation.identity
        else:
            return BasicQuantifierExpression(operation, index, constrain, body)


class BasicAtomicExpression(BasicExpression, AtomicExpression, ABC):
    def __init__(self, atom: Any, expression_type: Optional[ExpressionType] = None):
        if expression_type is None and not isinstance(atom, ExpressionType):
            # try to infer type for atom
            if isinstance(atom, Callable):
                expression_type = infer_python_callable_type(atom)
            else:
                expression_type = type(atom)
        super().__init__(expression_type)
        self._atom = atom

    @property
    def atom(self) -> str:
        return self._atom

    def syntactic_eq(self, other) -> bool:
        match other:
            case BasicAtomicExpression(base_type=other_base_type, atom=other_atom, type=other_type):
                return other_base_type == self.base_type and self.type == other_type and self.atom == other_atom
            case _:
                return False


class BasicVariable(BasicAtomicExpression, Variable):
    def __init__(self, name: str, type_: ExpressionType):
        if type_ is None:
            raise VariableNotTypedError
        BasicAtomicExpression.__init__(self, name, type_)


class BasicConstant(BasicAtomicExpression, Constant):
    def __init__(self, value: Any, type_: Optional[ExpressionType] = None):
        BasicAtomicExpression.__init__(self, value, type_)


class TrueContext(BasicConstant, Context):
    @property
    def dict(self) -> Dict[str, Any]:
        return {}

    @property
    def unsatisfiable(self) -> bool:
        return False

    @property
    def satisfiability_is_known(self) -> bool:
        return True

    def __init__(self):
        BasicConstant.__init__(self, True, bool)


class FalseContext(BasicConstant, Context):
    @property
    def dict(self) -> Dict[str, Any]:
        return {}

    @property
    def unsatisfiable(self) -> bool:
        return True

    @property
    def satisfiability_is_known(self) -> bool:
        return True

    def __init__(self):
        BasicConstant.__init__(self, False, bool)


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

    def syntactic_eq(self, other) -> bool:
        match other:
            case BasicFunctionApplication(subexpressions=other_subexpressions):
                return len(self.subexpressions) == len(other_subexpressions) and \
                       all(lhs.syntactic_eq(rhs) for lhs, rhs in zip(self.subexpressions, other_subexpressions))
            case _:
                return False


class BasicQuantifierExpression(QuantifierExpression, BasicExpression):
    def __init__(self, operation: AbelianOperation, index: Variable, constrain: Context, body: Expression):
        self._operation = operation
        self._index = index
        self._constrain = constrain
        self._body = body
        argument_types, return_type = get_args(operation.type)
        if len(argument_types) != 2 or not (argument_types[0] == argument_types [1] == return_type):
            raise ValueError(f"Wrong operation type {operation.type}.")
        super().__init__(return_type)

    @property
    def operation(self) -> Constant:
        return self._operation

    @property
    def index(self) -> Variable:
        return self._index

    @property
    def constrain(self) -> Expression:
        return self._constrain

    @property
    def body(self) -> Expression:
        return self._body

    def syntactic_eq(self, other) -> bool:
        match other:
            case BasicQuantifierExpression(subexpressions=other_subexpressions):
                return all(lhs.syntactic_eq(rhs) for lhs, rhs in zip(self.subexpressions, other_subexpressions))
            case _:
                return False


class BasicAbelianOperation(BasicConstant, AbelianOperation):
    @property
    def identity(self) -> Any:
        return self._identity

    def __init__(self, operation: Callable, element_type: type, identity: Any):
        # technically, the type of identity should be element_type, but there seems no way to declare that in Python?
        self._identity = identity
        super().__init__(operation, Callable[[element_type, element_type], element_type])


def basic_add_operation(type_) -> BasicAbelianOperation:
    return BasicAbelianOperation(operator.add, type_, 0)


class BasicSummation(BasicQuantifierExpression):
    def __init__(self, type_: type, index: Variable, constrain: Context, body: Expression):
        """
        Expect type_ to be the argument type/return type of the summation.
        E.g., the type_ of Summation{i in [0,100]}(i) should be `int`.
        """
        super().__init__(basic_add_operation(type_), index, constrain, body)
