from __future__ import annotations

from abc import ABC, abstractmethod
import fractions
from functools import cached_property, total_ordering
import typing
from typing import Any, Dict, Optional, Tuple, List, FrozenSet, Callable


import sympy
import z3

from neuralpp.symbolic.basic_expression import FalseContext, TrueContext
from neuralpp.symbolic.expression import (
    Constant,
    Context,
    Expression,
    FunctionApplication,
    Variable,
)
import neuralpp.symbolic.functions as functions
from neuralpp.util.callable_util import (
    ExpressionType,
    get_type_from_z3_object,
    type_to_z3_sort,
    python_callable_to_z3_function,
    z3_function_to_python_callable,
    apply_python_callable_on_z3_arguments,
)
from neuralpp.util.z3_util import (
    extract_key_to_value_from_assertions,
    is_z3_uninterpreted_function,
    is_z3_value,
    is_z3_variable,
    traverse_equalities,
    z3_add_solver_and_literal,
    z3_merge_solvers,
    z3_replace_in_solver,
)


def _z3_object_to_expression(z3_object: z3.ExprRef) -> Z3Expression:
    # Returns an instance of the appropriate subclass of Expression matching z3_object
    if z3_object.num_args() > 0:
        return Z3FunctionApplication(z3_object)
    elif (
        z3.is_int_value(z3_object)
        or z3.is_rational_value(z3_object)
        or z3.is_fp_value(z3_object)
        or z3.is_true(z3_object)
        or z3.is_false(z3_object)
    ):
        return Z3Constant(z3_object)
    else:
        return Z3Variable(z3_object)


class Z3Expression(Expression, ABC):
    @property
    @abstractmethod
    def z3_expression(self) -> z3.AstRef:
        pass

    @classmethod
    def new_constant(
        cls, value: Any, type_: Optional[ExpressionType] = None
    ) -> Z3Constant:
        if isinstance(value, z3.ExprRef | z3.FuncDeclRef):
            z3_object = value
        elif isinstance(value, bool):
            z3_object = z3.BoolVal(value)
        elif isinstance(value, int):
            z3_object = z3.IntVal(value)
        elif isinstance(value, float):
            # z3_object = z3.FPVal(value)
            z3_object = z3.RealVal(value)  # z3's fp is not supported yet
        elif isinstance(value, fractions.Fraction):
            z3_object = z3.RealVal(value)
        elif isinstance(value, str):
            if type_ is None:
                raise NotImplementedError(
                    "z3 requires specifying arguments and return "
                    "type of an uninterpreted function."
                )
            else:
                argument_types, return_type = typing.get_args(type_)
                z3_object = z3.Function(
                    value, *map(type_to_z3_sort, argument_types + [return_type])
                )
        else:
            if type_ is not None:
                # e.g., type_ = (int -> int -> int), then argument_type = int.
                argument_types = typing.get_args(type_)[0]
                argument_type = argument_types[0]
                if not all(typ == argument_type for typ in argument_types):
                    raise TypeError(
                        "Z3 expects all arguments to have the same type. If not, consider use"
                        "z3.ToInt()/z3.ToReal() to explicitly convert."
                    )
            else:
                argument_type = None
            z3_object = python_callable_to_z3_function(value, argument_type)
        return Z3Constant(z3_object)

    @classmethod
    def new_variable(cls, name: str, type_: ExpressionType) -> Z3Variable:
        if isinstance(type_, type(Callable[..., Any])):
            # isinstance(type_, Callable) is wrong: e.g., isinstance(int, Callable)==True
            argument_types, return_type = typing.get_args(type_)
            z3_var = z3.Function(name, *map(type_to_z3_sort, argument_types), type_to_z3_sort(return_type))
        else:
            z3_var = z3.Const(name, type_to_z3_sort(type_))
        return Z3Variable(z3_var)

    @classmethod
    def new_function_application(
        cls, function: Expression, arguments: List[Expression]
    ) -> Z3FunctionApplication:
        match function:
            case Z3Constant(z3_object=z3_function_declaration) | Z3Variable(
                z3_object=z3_function_declaration
            ):
                z3_arguments = [
                    cls.convert(argument).z3_expression for argument in arguments
                ]
                return Z3FunctionApplication(z3_function_declaration(*z3_arguments))
            case Constant(value=python_callable):
                z3_arguments = [
                    cls.convert(argument).z3_expression for argument in arguments
                ]
                return Z3FunctionApplication(
                    apply_python_callable_on_z3_arguments(
                        python_callable, *z3_arguments
                    )
                )
            case Variable(name=name):
                # TODO: this can be implemented without raising since we know the type of the uninterpreted function.
                raise ValueError(
                    f"Cannot create a Z3Expression from uninterpreted function {name}"
                )
            case FunctionApplication(_, _):
                raise ValueError("The function must be a python callable.")
            case _:
                raise ValueError(f"Unknown case: {function}")

    @classmethod
    def new_quantifier_expression(
        cls,
        operation: Constant,
        index: Variable,
        constraint: Expression,
        body: Expression,
        is_integral: bool,
    ) -> Expression:
        # Z3 can only represent "forall" and "exists" quantified expressions.
        raise NotImplementedError()

    @classmethod
    def pythonize_value(cls, value: z3.ExprRef | z3.FuncDeclRef) -> Any:
        if isinstance(value, z3.ExprRef):
            if value.sort() == z3.IntSort():
                return int(str(value))
            elif value.sort() == z3.FPSort(11, 53):
                return float(str(value))
            elif value.sort() == z3.RealSort():
                return fractions.Fraction(str(value))
            elif value.sort() == z3.BoolSort():
                return bool(value)
            else:
                raise TypeError(f"Unrecognized z3 sort {value.sort()}")
        elif isinstance(value, z3.FuncDeclRef):
            if is_z3_uninterpreted_function(value):
                return value.name()

            try:
                return z3_function_to_python_callable(value)
            except Exception as exc:
                if z3.is_to_real(value):
                    return value
                if value.kind() == z3.Z3_OP_TO_REAL:
                    return functions.identity
                raise ValueError(f"Cannot pythonize {value}.") from exc
        else:
            raise ValueError("Cannot pythonize non-z3 object")

    @classmethod
    def convert(cls, from_expression: Expression) -> Z3Expression:
        return cls._convert(from_expression)


class Z3ObjectExpression(Z3Expression, ABC):
    """
    Z3Expression that has a Z3 object, which is a symbolic expression.
    E.g.: Constants, FunctionApplication, etc.
    A Z3 solver is not a Z3 object
    """

    def __init__(self, z3_object: z3.ExprRef | z3.FuncDeclRef):
        Expression.__init__(self, get_type_from_z3_object(z3_object))
        self._z3_object = z3_object

    @property
    def z3_object(self):
        return self._z3_object

    @property
    def z3_expression(self) -> z3.AstRef:
        return self.z3_object

    def internal_object_eq(self, other) -> bool:
        match other:
            case Z3ObjectExpression(z3_object=other_z3_object):
                return self.z3_object.eq(other_z3_object)
            case _:
                return False

    def __hash__(self):
        return self.z3_object.__hash__()


class Z3Constant(Z3ObjectExpression, Constant):
    @property
    def atom(self) -> Any:
        return Z3Expression.pythonize_value(self._z3_object)


class Z3Variable(Z3ObjectExpression, Variable):
    @property
    def atom(self) -> str:
        return str(self._z3_object)


class Z3FunctionApplication(Z3ObjectExpression, FunctionApplication):
    """
    Note here z3_object cannot be z3.FuncDeclRef: add(1,2) is a ExprRef, while add is a FuncDeclRef.
    Z3 does not allow partial application nor high-order functions (arguments must be of atomic type such as int,
    not int -> int).
    """

    @property
    def function(self) -> Expression:
        if is_z3_uninterpreted_function(self._z3_object.decl()):
            return Z3Variable(self._z3_object.decl())
        else:
            return Z3Constant(self._z3_object.decl())

    @property
    def number_of_arguments(self) -> int:
        return self.z3_object.num_args()

    @property
    def arguments(self) -> List[Expression]:
        return list(map(_z3_object_to_expression, self._z3_object.children()))

    @property
    def native_arguments(self) -> Tuple[z3.AstRef, ...]:
        """faster than arguments"""
        return self.z3_object.children()  # z3 f.args returns a tuple

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.function] + self.arguments


class Z3SolverExpression(Context, Z3Expression, FunctionApplication):
    def __init__(
        self, z3_solver: z3.Solver = None, value_dict: Dict[str, Any] | None = None
    ):
        """
        Assume z3_solver is satisfiable, otherwise user should use Z3UnsatContext() instead.
        Also assumes value_dict is not contradictory to z3_solver. Formally, the following statement is valid:
            for all k,v in value_dict.item(), z3_solver.assertions implies k == v
        """
        super().__init__(bool)

        if z3_solver is None:
            z3_solver = z3.Solver()

        if not z3_solver.check() == z3.sat:
            raise ValueError(
                f"Expect a solver that is satisfiable. Got {z3_solver.check()}."
            )
        self._solver = z3_solver

        if value_dict is not None:
            self._dict = value_dict
        else:  # figure out ourselves
            self._dict = extract_key_to_value_from_assertions(z3_solver.assertions())

        match z3_solver.check():
            case z3.unsat:
                raise TypeError(
                    "Solver is unsatisfiable. Should use FalseContext() instead."
                )
            case z3.sat:
                self._unknown = False
            case z3.unknown:
                self._unknown = True

    @property
    def function(self) -> Expression:
        # conjunctive clause is an "and" of all arguments
        return Z3Constant(z3.And(True, True).decl())

    @property
    def number_of_arguments(self) -> int:
        return len(self.assertions)

    @cached_property
    def arguments(self) -> List[Expression]:
        return [_z3_object_to_expression(assertion) for assertion in self.assertions]

    @cached_property
    def assertions(self) -> z3.AstVector:
        return self._solver.assertions()

    @cached_property
    def z3_expression(self) -> z3.AstRef:
        return z3.And(self.assertions)

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.function] + self.arguments

    @property
    def dict(self) -> Dict[str, Any]:
        return self._dict

    @property
    def unsatisfiable(self) -> bool:  # self should always be satisfiable
        if not self._unknown:
            return False
        else:
            raise Context.UnknownError()

    @property
    def satisfiability_is_known(self) -> bool:
        return not self._unknown

    def replace(
        self, from_expression: Expression, to_expression: Expression
    ) -> Context:
        """
        If we do not override this method, the default implementation will cause the return value to be
        a Z3FunctionApplication, where the result is no longer a Context.
        """
        if self.syntactic_eq(from_expression):
            match to_expression:
                case Constant(value=True):
                    return TrueContext()
                case Constant(value=False):
                    return FalseContext()
                case _:
                    raise NotImplementedError(
                        "replace a SolverExpression with a non-constant value is not supported."
                    )

        from_expression = Z3Expression.convert(from_expression)
        to_expression = Z3Expression.convert(to_expression)

        if not isinstance(from_expression, Z3ObjectExpression):
            # only possible when from_expression is Z3SolverExpression
            # don't replace
            return self
        if not isinstance(to_expression, Z3ObjectExpression):
            raise ValueError(
                f"{to_expression}({type(to_expression)}) does not have a z3_object."
            )

        new_solver = z3_replace_in_solver(
            self._solver, from_expression.z3_object, to_expression.z3_object
        )
        return Z3SolverExpression(new_solver)

    def internal_object_eq(self, other) -> bool:
        return False  # why do we need to compare two Z3ConjunctiveClause?

    @staticmethod
    def make(solver: z3.Solver, dict_: Dict[str, Any]) -> Context:
        match solver.check():
            case z3.unsat:
                return FalseContext()
            case z3.sat | z3.unknown:
                return Z3SolverExpression(solver, dict_)

    @staticmethod
    def from_expression(expression: Expression) -> Z3SolverExpression:
        # a possible improvement is to split expression into subexpressions if it's an "And"
        assert isinstance(expression, Expression)
        if isinstance(expression, Z3SolverExpression):
            return expression

        if not isinstance(expression, Z3Expression):
            expression = Z3Expression.convert(expression)
        new_solver = z3.Solver()
        new_solver.add(expression.z3_object)
        return Z3SolverExpression(new_solver)

    @cached_property
    def variable_replacement_dict(self):
        """
        Since equality is transitive, a set of equalities would create a set of elements who are equal to each other.
        We refer to such a set as an "equivalence class". If we have a total order "<=" on the elements of the
        equivalence class, we can compute a minimum of that class. Note this total order "<=" has nothing to do with
        the equivalence relation.
        The elements in an equivalence class can be variables and at most one value (if two distinct value a and b
        exist in an equivalence class, then the context is unsatisfiable). A value is always smaller than all variables.

        A variable_replacement_dict is a (Variable | Expression | Value) -> (Variable | Expression | Value) dictionary
        that maps a variable in the context to the minimum of its equivalence class.
        E.g., suppose the total order of variables is the alphabetical order,
        and the context has the following assertions: { x == y, y > 1, z == y, a == 3, b == a },
        then variable_replacement_dict would be { x: x, y: x, z: x, a: 3, b: 3}.
        """
        return {
            _z3_object_to_expression(key): _z3_object_to_expression(value)
            for key, value in self.z3_variable_replacement_dict.items()
        }

    @cached_property
    def z3_variable_replacement_dict(self):
        """
        `variable_replacement_dict`, except in z3's native representation
        """
        result = {}
        equivalence_classes = (
            EquivalenceClass.extract_equivalence_classes_from_assertions(
                self.assertions
            )
        )
        for equivalence_class in equivalence_classes:
            for element in equivalence_class:
                result[
                    element.as_z3_expression()
                ] = equivalence_class.minimum.as_z3_expression()
        return result

    def __hash__(self):
        return self._solver.__hash__()

    def __and__(self, other: Any) -> Context:
        if self.unsatisfiable:
            # if an expression is unsatisfiable, its conjunction with anything is still unsatisfiable
            return self

        if isinstance(other, Z3SolverExpression):
            if other.unsatisfiable:
                return other
            new_solver = z3_merge_solvers(self._solver, other._solver)
            new_dict = self._dict | other._dict
        else:
            if not isinstance(other, Expression):
                other = Z3Expression.new_constant(other, None)
            elif not isinstance(other, Z3Expression):
                other = Z3Expression.convert(other)
            # Always treat `other` as a literal. Will raise if it cannot be converted to a boolean in Z3.
            new_solver = z3_add_solver_and_literal(self._solver, other.z3_object)
            new_dict = self._dict | extract_key_to_value_from_assertions(
                [other.z3_object]
            )
        return Z3SolverExpression.make(new_solver, new_dict)

    __rand__ = __and__


class Z3SolverExpressionDummy(Z3SolverExpression):
    """
    A Z3SolverExpression that knows nothing.
    """

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.function] + self.arguments

    @cached_property
    def z3_expression(self) -> z3.AstRef:
        return z3.BoolVal(True)

    @property
    def number_of_arguments(self) -> int:
        return 0

    @property
    def unsatisfiable(self) -> bool:  # self should always be satisfiable
        """Always False because I know nothing."""
        return False

    @property
    def satisfiability_is_known(self) -> bool:
        return True

    @staticmethod
    def from_expression(expression: Expression) -> Z3SolverExpression:
        raise NotImplementedError("?")

    def _is_known_to_imply_fastpath(self, expression: Expression) -> Optional[bool]:
        from .sympy_expression import SymPyExpression

        sympy_object = SymPyExpression.convert(expression).sympy_object
        if sympy_object == sympy.true:
            return True
        if sympy_object == sympy.false:
            return False

    def __and__(self, other: Any) -> Context:
        """I know nothing so I learn nothing."""
        return self

    __rand__ = __and__


@total_ordering
class OrderedZ3Expression:
    """
    OrderedZ3Expression helps determine what Expression should be used to replace the other
    equivalent expressions.
    In the case of a = b ^ b = c ^ a = 1, we would want to replace a, b, and c with 1.
    Constant < Variable < Expression
    E.g.: We will always want to replace a with 1.
    """

    def __init__(self, expr: z3.ExprRef):
        self._expr = expr

    @cached_property
    def expression_level(self):
        if is_z3_value(self._expr):
            return 0
        elif is_z3_variable(self._expr):
            return 1
        else:
            return 2

    def as_z3_expression(self):
        return self._expr

    def __eq__(self, other: OrderedZ3Expression) -> bool:
        if not isinstance(other, OrderedZ3Expression):
            return False
        return self._expr.eq(other._expr)

    def __hash__(self):
        return self._expr.__hash__()

    def __lt__(self, other: OrderedZ3Expression) -> bool:
        """
        Instead of creating a new symbolic expression, this `lt` returns a boolean
        """
        if not isinstance(other, OrderedZ3Expression):
            raise TypeError("Can only compare two OrderedZ3Expressions.")
        if self.expression_level < other.expression_level:
            return True
        elif other.expression_level < self.expression_level:
            return False
        else:  # same level
            return str(self._expr) < str(other._expr)


class EquivalenceClass:
    """
    Assume a = b ^ b = c ^ a = 1 ^ x = y. a = b = c = 1, which denotes a EquivalenceClass
    """

    def __init__(self, element: OrderedZ3Expression):
        self._set = frozenset([element])

    @cached_property
    def minimum(self) -> Any:
        """Assumes len(self) > 0."""
        return min(self._set)

    def __iter__(self):
        return self._set.__iter__()

    @staticmethod
    def union_and_update_both(class1: EquivalenceClass, class2: EquivalenceClass):
        """
        Given two equivalence classes, C1 and C2, first compute their union U = C1 U C2,
        then update both C1 and C2 to be U.
        """
        union = class1._set | class2._set
        class1._set = union
        class2._set = union

    @staticmethod
    def extract_equivalence_classes_from_assertions(
        assertions: List[z3.ExprRef] | z3.AstVector,
    ) -> FrozenSet[EquivalenceClass]:
        """
        Equivalence class is a set, see its definition in Z3SolverExpression.variable_replacement_dict.__doc__
        """

        def key_to_equivalence_class_accumulator(lhs: z3.ExprRef, rhs: z3.ExprRef):
            assert isinstance(lhs, z3.ExprRef)
            assert isinstance(rhs, z3.ExprRef)
            if lhs in result and rhs in result and result[lhs] == result[rhs]:
                pass  # they are already the same set, don't need to connect them
            else:
                if lhs not in result:
                    result[lhs] = EquivalenceClass(OrderedZ3Expression(lhs))
                if rhs not in result:
                    result[rhs] = EquivalenceClass(OrderedZ3Expression(rhs))
                EquivalenceClass.union_and_update_both(result[lhs], result[rhs])

        result: Dict[z3.ExprRef, EquivalenceClass] = {}
        traverse_equalities(assertions, key_to_equivalence_class_accumulator)
        return frozenset(result.values())

    def __eq__(
        self, other: FrozenSet
    ):  # either same instance or not equal, no need to use more expensive `==`.
        if not isinstance(other, EquivalenceClass):
            return False
        return self._set is other._set

    def __hash__(self):
        return self._set.__hash__()
