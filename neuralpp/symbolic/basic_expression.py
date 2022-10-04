from __future__ import annotations
import operator
from abc import ABC
from typing import List, Optional, Callable, Dict, get_args, Any

from neuralpp.symbolic.expression import (
    AbelianOperation,
    AtomicExpression,
    Constant,
    Context,
    Expression,
    FunctionApplication,
    QuantifierExpression,
    Variable,
)
from neuralpp.util.callable_util import ExpressionType, infer_python_callable_type
from neuralpp.util.symbolic_error_util import VariableNotTypedError


class BasicExpression(Expression, ABC):
    @classmethod
    def new_constant(
        cls, value: Any, type_: Optional[ExpressionType] = None
    ) -> BasicConstant:
        return BasicConstant(value, type_)

    @classmethod
    def new_variable(cls, name: str, type_: ExpressionType) -> BasicVariable:
        return BasicVariable(name, type_)

    @classmethod
    def new_function_application(
        cls, function: Expression, arguments: List[Expression]
    ) -> BasicFunctionApplication:
        return BasicFunctionApplication(function, arguments)

    @classmethod
    def new_quantifier_expression(
        cls,
        operation: AbelianOperation,
        index: Variable,
        constraint: Context,
        body: Expression,
        is_integral: bool,
    ) -> Expression:
        if constraint.satisfiability_is_known and constraint.unsatisfiable:
            return operation.identity
        else:
            return BasicQuantifierExpression(
                operation, index, constraint, body, is_integral
            )


class BasicAtomicExpression(BasicExpression, AtomicExpression, ABC):
    def __init__(self, atom: Any, expression_type: Optional[ExpressionType] = None):
        if expression_type is None and not isinstance(atom, ExpressionType):
            # try to infer type for atom
            if callable(atom):
                expression_type = infer_python_callable_type(atom)
            else:
                expression_type = type(atom)
        super().__init__(expression_type)

        self._atom = atom

    @property
    def atom(self) -> str:
        return self._atom

    def internal_object_eq(self, other) -> bool:
        match other:
            case BasicAtomicExpression(
                atom=other_atom, type=other_type
            ):
                return (
                    self.syntactic_form == other.syntactic_form
                    and self.type == other_type
                    and self.atom == other_atom
                )
            case _:
                return False

    def __hash__(self):
        return self.atom.__hash__()


class BasicConstant(BasicAtomicExpression, Constant):
    def __init__(self, value: Any, type_: Optional[ExpressionType] = None):
        BasicAtomicExpression.__init__(self, value, type_)


class BasicVariable(BasicAtomicExpression, Variable):
    def __init__(self, name: str, type_: ExpressionType):
        if type_ is None:
            raise VariableNotTypedError
        BasicAtomicExpression.__init__(self, name, type_)


class BasicFunctionApplication(BasicExpression, FunctionApplication):
    def __init__(self, function: Expression, arguments: List[Expression]):
        function_type = function.get_return_type(len(arguments))
        super().__init__(function_type)

        self._subexpressions = [function] + arguments
        self._arguments = self._subexpressions[1:]

    @property
    def function(self) -> Expression:
        return self._subexpressions[0]

    @property
    def number_of_arguments(self) -> int:
        return len(self.arguments)

    @property
    def arguments(self) -> List[Expression]:
        return self._arguments

    @property
    def subexpressions(self) -> List[Expression]:
        return self._subexpressions

    def internal_object_eq(self, other) -> bool:
        match other:
            case BasicFunctionApplication(subexpressions=other_subexpressions):
                return len(self.subexpressions) == len(other_subexpressions) and all(
                    lhs.internal_object_eq(rhs)
                    for lhs, rhs in zip(self.subexpressions, other_subexpressions)
                )
            case _:
                return False

    def __hash__(self):
        return hash(tuple(self.subexpressions))


class TrueContext(BasicConstant, Context):
    def __init__(self):
        BasicConstant.__init__(self, True, bool)

    @property
    def dict(self) -> Dict[str, Any]:
        return {}

    @property
    def unsatisfiable(self) -> bool:
        return False

    @property
    def satisfiability_is_known(self) -> bool:
        return True

    @property
    def and_priority(self) -> int:
        """
        Set and_priority to be higher than normal Context() to simplify __and__ operation.
        E.g., say we have a Z3SolverExpression z3_solver_expression:
        z3_solver_expression & TrueContext()
        would normally call
        z3_solver_expression.__and__(TrueContext())
        but by setting and_priority to 2 here, the above expression will call
        TrueContext().__and__(z3_solver_expression)
        which just returns the other side.
        """
        return 2

    def __and__(self, other: Any) -> Expression:
        return other

    __rand__ = __and__


class FalseContext(BasicConstant, Context):
    def __init__(self):
        BasicConstant.__init__(self, False, bool)

    @property
    def dict(self) -> Dict[str, Any]:
        return {}

    @property
    def unsatisfiable(self) -> bool:
        return True

    @property
    def satisfiability_is_known(self) -> bool:
        return True

    @property
    def and_priority(self) -> int:
        return 2

    def __and__(self, other: Any) -> Expression:
        return self

    __rand__ = __and__


class BasicAbelianOperation(BasicConstant, AbelianOperation):
    def __init__(
        self,
        operation: Callable,
        identity: Constant,
        inverse_function: Callable[[Expression], Expression],
    ):
        element_type = identity.type
        super().__init__(
            operation, Callable[[element_type, element_type], element_type]
        )

        # technically, the type of identity should be element_type, but there
        # seems no way to declare that in Python?
        self._identity = identity
        self._inverse_function = inverse_function

    @property
    def identity(self) -> Constant:
        assert isinstance(self._identity, Constant)
        return self._identity

    def inverse(self, expr: Expression) -> Expression:
        return self._inverse_function(expr)


class BasicQuantifierExpression(QuantifierExpression, BasicExpression):
    def __init__(
        self,
        operation: AbelianOperation,
        index: Variable,
        constraint: Context,
        body: Expression,
        is_integral: bool,
    ):
        argument_types, return_type = get_args(operation.type)
        if len(argument_types) != 2 or not (
            argument_types[0] == argument_types[1] == return_type
        ):
            raise ValueError(f"Wrong operation type {operation.type}.")

        super().__init__(return_type)

        self._operation = operation
        self._index = index
        self._constraint = constraint
        self._body = body
        self._is_integral = is_integral

    @property
    def operation(self) -> Constant:
        assert isinstance(self._operation, Constant)
        return self._operation

    @property
    def index(self) -> Variable:
        assert isinstance(self._index, Variable)
        return self._index

    @property
    def constraint(self) -> Expression:
        assert isinstance(self._constraint, Context)
        return self._constraint

    @property
    def body(self) -> Expression:
        assert isinstance(self._body, Expression)
        return self._body

    @property
    def is_integral(self) -> bool:
        return self._is_integral

    def internal_object_eq(self, other) -> bool:
        match other:
            case BasicQuantifierExpression(
                subexpressions=other_subexpressions, is_integral=other_is_integral
            ):
                return self.is_integral == other_is_integral and all(
                    lhs.internal_object_eq(rhs)
                    for lhs, rhs in zip(self.subexpressions, other_subexpressions)
                )
            case _:
                return False

    def __hash__(self):
        return hash(tuple(self.subexpressions))


def basic_add_operation(type_) -> BasicAbelianOperation:
    return BasicAbelianOperation(operator.add, BasicConstant(0, type_), operator.neg)


def basic_summation(
    type_: type, index: Variable, constraint: Context, body: Expression
) -> BasicQuantifierExpression:
    """
    Expect type_ to be the argument type/return type of the summation.
    E.g., the type_ of Summation{i in [0,100]}(i) should be `int`.
    """
    return BasicQuantifierExpression(
        basic_add_operation(type_), index, constraint, body, False
    )


def basic_integral(
    index: Variable, constraint: Context, body: Expression
) -> BasicQuantifierExpression:
    # basic_add_operation.element_type is redundant here
    return BasicQuantifierExpression(
        basic_add_operation(int), index, constraint, body, True
    )
