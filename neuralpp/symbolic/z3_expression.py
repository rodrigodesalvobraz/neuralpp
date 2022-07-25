from __future__ import annotations

import fractions
import typing
from typing import List, Any, Callable, Tuple, Optional, Type, Dict, FrozenSet
from abc import ABC, abstractmethod

import z3
import operator
import builtins
from neuralpp.symbolic.expression import Expression, FunctionApplication, Variable, Constant, ExpressionType, Context, \
    QuantifierExpression
from neuralpp.symbolic.basic_expression import FalseContext
from neuralpp.util.z3_util import z3_merge_solvers, z3_add_solver_and_literal, is_z3_uninterpreted_function, \
    z3_replace_in_solver
from functools import cached_property, total_ordering
import neuralpp.symbolic.functions as functions


def _get_type_from_z3_object(z3_object: z3.ExprRef | z3.FuncDeclRef) -> ExpressionType:
    """
    Z3 uses the word 'sort' in a similar sense to 'type': e.g., z3.IntSort() and z3.BoolSort().
    z3.ArraySort(z3.IntSort(), z3.IntSort()) is the type of arrays who's indexed by int and whose elements are int.
    However, 'int -> int' is not a sort.
    """
    match z3_object:
        case z3.FuncDeclRef():  # for uninterpreted functions.
            return Callable[[_z3_sort_to_type(z3_object.domain(i)) for i in range(z3_object.arity())],
                            _z3_sort_to_type(z3_object.range())]
        case z3.ExprRef():
            # this can be an atom such as "x:int" or "2", or this can also be "1+2".
            # Either way, the type is the return type, so just sort().
            return _z3_sort_to_type(z3_object.sort())


sort_type_relation = [
    (z3.IntSort(), int),
    (z3.BoolSort(), bool),
    (z3.RealSort(), fractions.Fraction),
    # (z3.FPSort(11, 53), float)  # FPSort(11,53) is double sort (IEEE754, ebits=11, sbits=53)
    # please refer to test/quick_tests/symbolic/z3_usage_test.py:test_z3_fp_sort() for why z3 floating point is not yet
    # supported
]
sort_type_dict = {sort: type_ for sort, type_ in sort_type_relation}
type_sort_dict = {type_: sort for sort, type_ in sort_type_relation}


def _z3_sort_to_type(sort: z3.Sort) -> Type:
    if sort in sort_type_dict:
        return sort_type_dict[sort]
    else:
        raise TypeError(f"Unrecognized z3 sort {sort}")


def _type_to_z3_sort(type_: Type) -> z3.Sort:
    if type_ in type_sort_dict:
        return type_sort_dict[type_]
    else:
        raise TypeError(f"Unrecognized type {type_}")


def _python_callable_to_z3_function(python_callable: Callable, type_: Optional[ExpressionType] = None) -> z3.FuncDeclRef:
    """
    Note: type_ is *not* the function type of the python_callable, but just the type of argument(s).
    This is not ambiguous because z3 does not allow the arguments (of the listed functions here) to be different,
    e.g., "add: float -> int -> int" is not accepted in z3, nor is "less_than: rational -> int -> bool".
    So the arguments must be of the same type.
    E.g., if python_callable is operator.add whose function type is "int -> int -> int", then type_ should be "int".
    """
    match python_callable:
        # boolean operation
        case operator.and_:
            # We do a bit hack here because z3py does not provide direct access to e.g., "add function"
            # So we have to create a function application in z3 and retrieve its declaration using decl().
            return z3.And(True, True).decl()
        case operator.or_:
            return z3.Or(True, True).decl()
        case operator.invert:
            return z3.Not(True).decl()
        case operator.xor:
            return z3.Xor(True, True).decl()

    x, y = z3.Consts("x y", _type_to_z3_sort(type_) if type_ is not None else z3.IntSort())
    match python_callable:
        # comparison and arithmetic are overloaded by z3.
        case operator.le:
            return (x <= y).decl()
        case operator.lt:
            return (x < y).decl()
        case operator.ge:
            return (x >= y).decl()
        case operator.gt:
            return (x > y).decl()
        case operator.eq:
            return (x == y).decl()
        case operator.add:
            return (x + y).decl()
        case operator.sub:
            return (x - y).decl()
        case operator.neg:
            return (-x).decl()
        case operator.mul:
            return (x * y).decl()
        case operator.pow:
            return (x ** y).decl()
        # min/max
        case builtins.min:
            raise NotImplementedError("Cannot convert min to a z3 function declaration."
                                      "However we can create z3.If(x<y, x, y) for min(x,y).")
        case builtins.max:
            # if len(arguments) != 2:
            #     raise NotImplementedError("Only 2-element max is supported")
            # return z3.If(arguments[0] > arguments[1], arguments[0], arguments[1])
            raise NotImplementedError("Cannot convert min to a z3 function declaration."
                                      "However we can create z3.If(x>y, x, y) for max(x,y).")
        # if then else
        case functions.conditional:
            return z3.If(x > y, x, y).decl()  # "x>y" is just a placeholder boolean.
        case _:
            raise ValueError(f"Python callable {python_callable} is not recognized.")


def _z3_function_to_python_callable(z3_function: z3.FuncDeclRef) -> Callable:
    match z3_function.kind():
        # boolean operation
        case z3.Z3_OP_AND:
            return operator.and_
        case z3.Z3_OP_OR:
            return operator.or_
        case z3.Z3_OP_NOT:
            return operator.invert
        case z3.Z3_OP_XOR:
            return operator.xor
        # comparison
        case z3.Z3_OP_LE:
            return operator.le
        case z3.Z3_OP_LT:
            return operator.lt
        case z3.Z3_OP_GE:
            return operator.ge
        case z3.Z3_OP_GT:
            return operator.gt
        case z3.Z3_OP_EQ:
            return operator.eq
        # arithmetic
        case z3.Z3_OP_ADD:
            return operator.add
        case z3.Z3_OP_SUB:
            return operator.sub
        case z3.Z3_OP_UMINUS:
            return operator.neg
        case z3.Z3_OP_MUL:
            return operator.mul
        case z3.Z3_OP_POWER:
            return operator.pow
        # if then else
        case z3.Z3_OP_ITE:
            return functions.conditional
        case _:
            raise ValueError(f"Z3 function {z3_function} is not recognized.")


def _z3_object_to_expression(z3_object: z3.ExprRef) -> Z3Expression:
    # Returns an instance of the appropriate subclass of Expression matching z3_object
    if z3_object.num_args() > 0:
        return Z3FunctionApplication(z3_object)
    elif z3.is_int_value(z3_object) or z3.is_rational_value(z3_object) or z3.is_fp_value(z3_object) or \
            z3.is_true(z3_object) or z3.is_false(z3_object):
        return Z3Constant(z3_object)
    else:
        return Z3Variable(z3_object)


# On the type of *arguments:
# https://peps.python.org/pep-0484/#arbitrary-argument-lists-and-default-argument-values
def _apply_python_callable_on_z3_arguments(python_callable: Callable,
                                           *arguments: z3.BoolRef | z3.ArithRef) -> z3.ExprRef:
    """
    Directly calling this function can do something that _python_callable_to_z3_function(python_callable)(arguments)
    cannot do:
        `_python_callable_to_z3_function(builtins.min)(x, y)` raises an error, because "min" cannot be turned
        into a z3 function.
    while
        `_apply_python_callable_on_z3_arguments(builtins.min, x, y)` is fine, because "min(x,y)" can be turned
        into a z3 function application (namely, If(x<y, x, y))
    """
    match python_callable:
        # boolean operation
        case operator.and_:
            return z3.And(arguments)
        case operator.or_:
            return z3.Or(arguments)
        case operator.invert:
            return z3.Not(arguments[0])
        case operator.xor:
            return z3.Xor(arguments[0], arguments[1])
        # comparison
        case operator.le:
            return arguments[0] <= arguments[1]
        case operator.lt:
            return arguments[0] < arguments[1]
        case operator.ge:
            return arguments[0] >= arguments[1]
        case operator.gt:
            return arguments[0] > arguments[1]
        case operator.eq:
            return arguments[0] == arguments[1]
        # arithmetic
        case operator.add:
            return arguments[0] + arguments[1]
        case operator.sub:
            return arguments[0] - arguments[1]
        case operator.neg:
            return - arguments[0]
        case operator.mul:
            return arguments[0] * arguments[1]
        case operator.pow:
            return arguments[0] ** arguments[1]
        # min/max
        case builtins.min:
            return z3.If(arguments[0] < arguments[1], arguments[0], arguments[1])
        case builtins.max:
            return z3.If(arguments[0] > arguments[1], arguments[0], arguments[1])
        # our functions
        case functions.conditional:
            return z3.If(arguments[0], arguments[1], arguments[2])
        case _:
            raise ValueError(f"Python callable {python_callable} is not recognized.")


class Z3Expression(Expression, ABC):
    @classmethod
    def new_constant(cls, value: Any, type_: Optional[ExpressionType] = None) -> Z3Constant:
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
                raise NotImplementedError("z3 requires specifying arguments and return "
                                          "type of an uninterpreted function.")
            else:
                argument_types, return_type = typing.get_args(type_)
                z3_object = z3.Function(value, *map(_type_to_z3_sort, argument_types + [return_type]))
        else:
            if type_ is not None:
                # e.g., type_ = (int -> int -> int), then argument_type = int.
                argument_types = typing.get_args(type_)[0]
                argument_type = argument_types[0]
                if not all(typ == argument_type for typ in argument_types):
                    raise TypeError("Z3 expects all arguments to have the same type. If not, consider use"
                                    "z3.ToInt()/z3.ToReal() to explicitly convert.")
            else:
                argument_type = None
            z3_object = _python_callable_to_z3_function(value, argument_type)
        return Z3Constant(z3_object)

    @classmethod
    def new_variable(cls, name: str, type_: ExpressionType) -> Z3Variable:
        z3_var = z3.Const(name, _type_to_z3_sort(type_))
        return Z3Variable(z3_var)

    @classmethod
    def new_function_application(cls, function: Expression, arguments: List[Expression]) -> Z3FunctionApplication:
        match function:
            case Z3Constant(z3_object=z3_function_declaration) | Z3Variable(z3_object=z3_function_declaration):
                z3_arguments = [cls.convert(argument).z3_expression for argument in arguments]
                return Z3FunctionApplication(z3_function_declaration(*z3_arguments))
            case Constant(value=python_callable):
                z3_arguments = [cls.convert(argument).z3_expression for argument in arguments]
                return Z3FunctionApplication(_apply_python_callable_on_z3_arguments(python_callable, *z3_arguments))
            case Variable(name=name):
                # TODO: this can be implemented without raising since we know the type of the uninterpreted function.
                raise ValueError(f"Cannot create a Z3Expression from uninterpreted function {name}")
            case FunctionApplication(_, _):
                raise ValueError("The function must be a python callable.")
            case _:
                raise ValueError(f"Unknown case: {function}")

    @classmethod
    def new_quantifier_expression(cls, operation: Constant, index: Variable, constraint: Expression, body: Expression,
                                  ) -> Expression:
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
                return _z3_function_to_python_callable(value)
            except Exception:
                raise ValueError(f"Cannot pythonize {value}.")
        else:
            raise ValueError("Cannot pythonize non-z3 object")

    @classmethod
    def convert(cls, from_expression: Expression) -> Z3Expression:
        return cls._convert(from_expression)

    @property
    @abstractmethod
    def z3_expression(self) -> z3.AstRef:
        pass


class Z3ObjectExpression(Z3Expression, ABC):
    def __init__(self, z3_object: z3.ExprRef | z3.FuncDeclRef):
        Expression.__init__(self, _get_type_from_z3_object(z3_object))
        self._z3_object = z3_object

    @property
    def z3_object(self):
        return self._z3_object

    def syntactic_eq(self, other) -> bool:
        match other:
            case Z3ObjectExpression(z3_object=other_z3_object):
                return self.z3_object.eq(other_z3_object)
            case _:
                return False

    @property
    def z3_expression(self) -> z3.AstRef:
        return self.z3_object

    def __hash__(self):
        return self.z3_object.__hash__()


class Z3Variable(Z3ObjectExpression, Variable):
    @property
    def atom(self) -> str:
        return str(self._z3_object)


class Z3Constant(Z3ObjectExpression, Constant):
    @property
    def atom(self) -> Any:
        return Z3Expression.pythonize_value(self._z3_object)


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
    def arguments(self) -> List[Expression]:
        return list(map(_z3_object_to_expression, self._z3_object.children()))

    @property
    def native_arguments(self) -> Tuple[z3.AstRef, ...]:
        """ faster than arguments """
        return self.z3_object.children()  # z3 f.args returns a tuple

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.function] + self.arguments

    @property
    def number_of_arguments(self) -> int:
        return self.z3_object.num_args()


z3_false = z3.BoolVal(False)


def _traverse_equalities(assertions: List[z3.ExprRef], accumulator: Callable[[z3.ExprRef, z3.ExprRef], type(None)]):
    """
    A high-order function that takes a function accumulator as an argument.
    Traverse all the equalities that are leaves of conjunctions and apply accumulator on each of them.
    The accumulator takes the both sides of the equalities as two of its arguments (with lhs comes first).
    """
    for assertion in assertions:
        if z3.is_eq(assertion):
            accumulator(assertion.arg(0), assertion.arg(1))
        elif z3.is_and(assertion):
            _traverse_equalities(assertion.children(), accumulator)


def _is_z3_value(v):
    return z3.is_int_value(v) or z3.is_rational_value(v) or z3.is_algebraic_value(v) or z3.is_fp_value(v)


def _is_z3_variable(k):
    return z3.is_const(k) and not _is_z3_value(k)


def _extract_key_to_value_from_assertions(assertions: List[z3.ExprRef] | z3.AstVector) -> Dict[str, Any]:
    """
    Extract all key-value pairs if it's in the form of "k1 == v1 & k2 == v2 & ..". `And` can be nested:
    we can extract all 4 pairs in And(k1 == v1, k2 == v2, And(k3 == v3, k4 == v4)).
    """
    def key_to_value_accumulator(lhs: z3.ExprRef, rhs: z3.ExprRef):
        if _is_z3_variable(lhs) and _is_z3_value(rhs):
            result[str(lhs)] = rhs
        elif _is_z3_variable(rhs) and _is_z3_value(lhs):
            result[str(rhs)] = lhs
    result = {}
    _traverse_equalities(assertions, key_to_value_accumulator)
    return result


class Z3SolverExpression(Context, Z3Expression, FunctionApplication):
    @property
    def dict(self) -> Dict[str, Any]:
        return self._dict

    def syntactic_eq(self, other) -> bool:
        return False  # why do we need to compare two Z3ConjunctiveClause?

    def replace(self, from_expression: Expression, to_expression: Expression) -> Z3SolverExpression:
        """
        If we do not override this replace(), the default replace() will cause the return value to be
        a Z3FunctionApplication, where the result is no longer a Context.
        """
        from_expression = Z3Expression.convert(from_expression)
        to_expression = Z3Expression.convert(to_expression)

        if not isinstance(from_expression, Z3ObjectExpression):
            # only possible when from_expression is Z3SolverExpression
            raise ValueError(f"{from_expression}({type(from_expression)}) does not have a z3_object.")
        if not isinstance(to_expression, Z3ObjectExpression):
            raise ValueError(f"{to_expression}({type(to_expression)}) does not have a z3_object.")

        new_solver = z3_replace_in_solver(self._solver, from_expression.z3_object, to_expression.z3_object)
        return Z3SolverExpression(new_solver)

    def __init__(self, z3_solver: z3.Solver = None, value_dict: Dict[str, Any] | None = None):
        """
        Assume z3_solver is satisfiable, otherwise user should use Z3UnsatContext() instead.
        Also assumes value_dict is not contradictory to z3_solver. Formally, the following statement is valid:
            for all k,v in value_dict.item(), z3_solver.assertions implies k == v
        """
        if z3_solver is None:
            z3_solver = z3.Solver()

        if not z3_solver.check() == z3.sat:
            raise ValueError(f"Expect a solver that is satisfiable. Got {z3_solver.check()}.")

        super().__init__(z3_false)
        self._solver = z3_solver
        if value_dict is not None:
            self._dict = value_dict
        else:  # figure out ourselves
            self._dict = _extract_key_to_value_from_assertions(z3_solver.assertions())

        match z3_solver.check():
            case z3.unsat:
                raise TypeError("Solver is unsatisfiable. Should use FalseContext() instead.")
            case z3.sat:
                self._unknown = False
            case z3.unknown:
                self._unknown = True

    @cached_property
    def assertions(self) -> z3.AstVector:
        return self._solver.assertions()

    @property
    def function(self) -> Expression:
        # conjunctive clause is an "and" of all arguments
        return Z3Constant(z3.And(True, True).decl())

    @cached_property
    def arguments(self) -> List[Expression]:
        return [_z3_object_to_expression(assertion) for assertion in self.assertions]

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.function] + self.arguments

    @property
    def unsatisfiable(self) -> bool:  # self should always be satisfiable
        if not self._unknown:
            return False
        else:
            raise Context.UnknownError()

    @property
    def satisfiability_is_known(self) -> bool:
        return not self._unknown

    @property
    def number_of_arguments(self) -> int:
        return len(self.assertions)

    @cached_property
    def z3_expression(self) -> z3.AstRef:
        return z3.And(self.assertions)

    @staticmethod
    def make(solver: z3.Solver, dict_: Dict[str, Any]) -> Context:
        match solver.check():
            case z3.unsat:
                return FalseContext()
            case z3.sat | z3.unknown:
                return Z3SolverExpression(solver, dict_)

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
            new_dict = self._dict | _extract_key_to_value_from_assertions([other.z3_object])
        return Z3SolverExpression.make(new_solver, new_dict)

    __rand__ = __and__

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
        return {_z3_object_to_expression(key): _z3_object_to_expression(value)
                for key, value in self.z3_variable_replacement_dict.items()}

    @cached_property
    def z3_variable_replacement_dict(self):
        """
        `variable_replacement_dict`, except in z3's native representation
        """
        result = {}
        equivalence_classes = _extract_equivalence_classes_from_assertions(self.assertions)
        for equivalence_class in equivalence_classes:
            for element in equivalence_class:
                result[element.as_z3_expression()] = equivalence_class.minimum.as_z3_expression()
        return result


class EquivalenceClass:
    def __init__(self, element: OrderedZ3Expression):
        self._set = frozenset([element])

    def __eq__(self, other: FrozenSet):  # either same instance or not equal, no need to use more expensive `==`.
        if not isinstance(other, EquivalenceClass):
            return False
        return self._set is other._set

    def __hash__(self):
        return self._set.__hash__()

    @cached_property
    def minimum(self) -> Any:
        """ Assumes len(self) > 0. """
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


@total_ordering
class OrderedZ3Expression:
    @cached_property
    def expression_level(self):
        if _is_z3_value(self._expr):
            return 0
        elif _is_z3_variable(self._expr):
            return 1
        else:
            return 2

    def as_z3_expression(self):
        return self._expr

    def __init__(self, expr: z3.ExprRef):
        self._expr = expr

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


def _extract_equivalence_classes_from_assertions(assertions: List[z3.ExprRef] | z3.AstVector) \
        -> FrozenSet[EquivalenceClass]:
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
    _traverse_equalities(assertions, key_to_equivalence_class_accumulator)
    return frozenset(result.values())


