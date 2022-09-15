from __future__ import (
    annotations,
)  # to support forward reference for recursive type reference

import operator
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Callable, Dict

from neuralpp.util.callable_util import (
    ExpressionType,
    get_arithmetic_function_type_from_argument_types,
    return_type_after_application,
    get_comparison_function_type_from_argument_types,
)
from neuralpp.util.util import same_len_and_predicate_true_for_all_pairs


class Expression(ABC):
    """
    Expression is the main interface for all symbolic expressions in the library.
    It is designed to integrate multiple _backends_ for symbolic computation
    such as SymPy, Z3 etc through implementations based on such backends.

    Currently, all Expressions must be of one of four _syntactic forms_:
    Constant, Variable, FunctionApplication, and QuantifierExpression.
    While those are subinterfaces of Expression, not all implementations
    of interface must necessarily implement one of those, but all implementations
    must have properties `form` and `form_kind` informing which
    interface a current instance satisfies.
    """
    def __init__(self, expression_type: ExpressionType):
        self._type = expression_type

    @property
    def form(self) -> Expression:
        """
        Syntactic form of expression.
        It consists of an expression that is an instance of one of the main interfaces for Expressions:
        Constant, Variable, FunctionApplication, QuantifiedExpression, etc.

        For expressions that are already instances of such interfaces, it suffices to return self,
        so this is the default implementation.

        For other subclasses of Expression that may be of varying forms
        (for example, instances of a class implementing a polynomial may have Constant, Variable or FunctionApplication
        forms, depending on the specific polynomial),
        the method must return an instance of one of these interfaces, with the appropriate attributes.

        One possible implementation in such cases is to return an instance of Basic* classes for
        the appropriate form, containing the appropriate sub-expressions.
        For example, polynomial "x" would return BasicVariable("x")
        while "x^2" would return a BasicFunctionApplication with the appropriate sub-expressions.

        Note that for algorithms using match ... case to be
        more generally applicable (beyond instances of the form interfaces to unforeseen Expression subclasses),
        they must match expressions `form` property rather than the expression themselves:
        match expression.form:
            case FunctionApplication(function=...)
        """
        return self

    @property
    def form_kind(self) -> type[Expression]:
        """
        The interface among Constant, Variable, FunctionApplication and QuantifierExpression
        that self.form instantiates.
        Note that this is not the same as type(self.form), which evaluates to a subclass of self.form_kind.
        This is why we call it form *kind* rather than form *type*.
        """
        raise NotImplementedError

    @property
    def is_polynomial(self) -> bool:
        return False

    @property
    @abstractmethod
    def subexpressions(self) -> List[Expression]:
        """
        Returns a list of subexpressions.
        E.g., subexpressions(f(x,y)) = [f,x,y]
        """
        pass

    @property
    def type(self) -> ExpressionType:
        return self._type

    @property
    def and_priority(self) -> int:
        """
        This property is by default set to 0 and indicates the priority of
        an expression being the first element in a conjunction
        (higher values indicate higher priority).
        For example: if a and b are Expressions, both having __and__() overloaded:
        >>> a: Expression
        >>> b: Expression
        then
        >>> a & b
        would mean
        >>> a.__and__(b)
        However, if b is a subclass that overload `and_property` to a > 0 value, then
        >>> a & b
        would mean
        >>> b.__and__(a)

        In particualr, this is useful in `Context`, since we want `Context` object to always `overshadow` its neighbors.
        So that
        literal & context
        would cause
        context.__and__(literal)
        thus adding literal to the context (instead of creating a new expression where we lost the context information).
        """
        return 0

    @abstractmethod
    def set(self, i: int, new_expression: Expression) -> Expression:
        """
        Set i-th subexpressions to new_expression. Count from 0.
        E.g., f(x,y).set(1,z) = [f,z,y].
        If it's out of the scope, return error.
        """
        pass

    @abstractmethod
    def replace(
            self, from_expression: Expression, to_expression: Expression
    ) -> Expression:
        """
        Every expression is immutable so replace() returns either self or a new Expression.
        No in-place modification should be made.
        If from_expression is not in the expression, returns self.
        """
        pass

    def contains(self, target: Expression) -> bool:
        """
        Checks if `target` is contained in `self`. The check is deep. An expression contains itself.
        E.g., f(x,f(a,b)).contains(a) == True and a.contains(a) == True
        """
        if self.syntactic_eq(target):
            return True
        for sub_expr in self.subexpressions:
            if sub_expr.contains(target):
                return True
        return False

    @abstractmethod
    def internal_object_eq(self, other) -> bool:
        """
        Returns True if self and other are instances of Expression and their internal representation are equal.
        This method usually depends on subclass-specific library calls,
        E.g., Z3Expression.internal_object_eq() would leverage z3.eq().
        This method should be considered as a cheap way to check internal object equality of two symbolic expressions
        known to be instances of the same class, as opposed to expressions represented in different
        frameworks whose comparison requires conversion.
        """
        pass

    def syntactic_eq(self, other) -> bool:
        """
        Returns if self and other are syntactically equivalent, i.e., that they have the same Expression interfaces.
        E.g, a Z3Expression of "a + b" is not considered equal to a SymPyExpression of "a + b" by internal_object_eq(),
        but is considered equal to it by syntactic_eq().
        """
        from neuralpp.symbolic.sympy_expression import SymPyPoly
        if self.internal_object_eq(other):
            return True

        match self.form, other.form:
            case AtomicExpression(atom=self_atom), AtomicExpression(atom=other_atom):
                # TODO: once SymPyExpression typing system is fixed, include `self_type == other_type`
                return self.form_kind == other.form_kind and self_atom == other_atom
            case _:  # anything not atomic
                return self.form_kind == other.form_kind and \
                       same_len_and_predicate_true_for_all_pairs(
                           self.form.subexpressions, other.form.subexpressions, Expression.syntactic_eq)

    @classmethod
    @abstractmethod
    def new_constant(cls, value: Any, type_: Optional[ExpressionType]) -> Constant:
        """
        Value is expected to be a python object or a "native" object.
        E.g., SymPyExpression.new_constant()'s legal input would have python `int` and `sympy.Integer`,
        but not `z3.Int`. Similarly, Z3Expression.new_constant()'s legal input includes `int` and `z3.Int` but
        not `sympy.Integer`.
        """
        pass

    @classmethod
    @abstractmethod
    def new_variable(cls, name: str, type_: ExpressionType) -> Variable:
        pass

    @classmethod
    @abstractmethod
    def new_function_application(
            cls, function: Expression, arguments: List[Expression]
    ) -> Expression:
        pass

    @classmethod
    @abstractmethod
    def new_quantifier_expression(
            cls,
            operation: Constant,
            index: Variable,
            constraint: Expression,
            body: Expression,
            is_integral: bool,
    ) -> Expression:
        pass

    @classmethod
    def _convert(cls, from_expression: Expression) -> Expression:
        """General helper for converting an Expression into this subclass of Expression."""
        if isinstance(from_expression, cls):
            return from_expression
        match from_expression.form:
            case Constant(value=value, type=type_):
                return cls.new_constant(value, type_)
            case Variable(name=name, type=type_):
                return cls.new_variable(name, type_)
            case FunctionApplication(function=function, arguments=arguments):
                return cls.new_function_application(function, arguments)
            case QuantifierExpression(
                subexpressions=subexpressions, is_integral=is_integral
            ):
                return cls.new_quantifier_expression(
                    *subexpressions, is_integral=is_integral
                )
            case _:
                raise ValueError(
                    f"invalid from_expression {from_expression}: {type(from_expression)}"
                )

    def get_return_type(self, number_of_arguments: int) -> ExpressionType:
        if not isinstance(self.form.type, type(Callable[..., Any])):
            raise TypeError(f"{self}'s function is not of function type.")
        return return_type_after_application(self.form.type, number_of_arguments)

    # ##### Methods for operator overloading

    def _new_binary_arithmetic(
            self, other, operator_, function_type=None, reverse=False
    ) -> Expression:
        return self._new_binary_operation(
            other, operator_, function_type, reverse, arithmetic=True
        )

    def _new_binary_boolean(self, other, operator_, reverse=False) -> Expression:
        return self._new_binary_operation(
            other, operator_, Callable[[bool, bool], bool], reverse, arithmetic=False
        )

    def _new_binary_comparison(
            self, other, operator_, function_type=None, reverse=False
    ) -> Expression:
        return self._new_binary_operation(
            other,
            operator_,
            function_type,
            reverse,
            arithmetic=False,
            arithmetic_arguments=True,
        )

    def _new_binary_operation(
            self,
            other,
            operator_,
            function_type=None,
            reverse=False,
            arithmetic=True,
            arithmetic_arguments=False,
    ) -> Expression:
        """
        Basic method for making a binary operation using the same type backend as `self`.
        Tries to convert `other` to a Constant if it is not an Expression.
        E.g., if operator_ is `+`, other is `3`. return self + Constant(3).
        By default, self is the 1st argument and `other` is the 2nd.
        If reverse is set to True, it is reversed, so e.g., if operator_ is `-` and reverse is True,
        the `other - self` is returned instead.
        If `arithmetic` is True, the return type is inferred from the argument types. Otherwise, it's assumed to
        be bool.
        """
        if not isinstance(other, Expression):
            other = self.new_constant(other, None)
        arguments = [self, other] if not reverse else [other, self]
        if function_type is None:
            if arithmetic:
                function_type = get_arithmetic_function_type_from_argument_types(
                    [arguments[0].type, arguments[1].type]
                )
            else:
                if arguments[0].type != arguments[1].type:
                    if arithmetic_arguments:
                        function_type = (
                            get_comparison_function_type_from_argument_types(
                                [arguments[0].type, arguments[1].type]
                            )
                        )
                    else:
                        raise TypeError(
                            f"Argument types mismatch: {arguments[0].type} != {arguments[1].type}. "
                            f"{arguments[0]}, {arguments[1]}"
                        )
                else:
                    function_type = Callable[
                        [arguments[0].type, arguments[1].type], bool
                    ]
        return self.new_function_application(
            self.new_constant(operator_, function_type), arguments
        )

    def __add__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.add)

    def __radd__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.add, reverse=True)

    def __mul__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.mul)

    def __rmul__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.mul, reverse=True)

    def __pow__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.pow)

    def __rpow__(self, other) -> Expression:
        return self._new_binary_arithmetic(other, operator.pow, reverse=True)

    def __truediv__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.truediv, reverse=True)

    def __sub__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.sub)

    def __rsub__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.sub, reverse=True)

    def __neg__(self) -> Expression:
        return self.new_function_application(
            self.new_constant(operator.neg, Callable[[self.type], self.type]), [self]
        )

    def __and__(self, other: Any) -> Expression:
        if isinstance(other, Expression) and other.and_priority > self.and_priority:
            return other.__and__(self)
        else:
            return self._new_binary_boolean(other, operator.and_)

    def __rand__(self, other: Any) -> Expression:
        return self._new_binary_boolean(other, operator.and_, reverse=True)

    def __or__(self, other: Any) -> Expression:
        return self._new_binary_boolean(other, operator.or_)

    def __ror__(self, other: Any) -> Expression:
        return self._new_binary_boolean(other, operator.or_, reverse=True)

    def __invert__(self) -> Expression:
        return self.new_function_application(
            self.new_constant(operator.invert, Callable[[bool], bool]), [self]
        )

    def __lt__(self, other) -> Expression:
        return self._new_binary_comparison(other, operator.lt)

    def __le__(self, other) -> Expression:
        return self._new_binary_comparison(other, operator.le)

    def __gt__(self, other) -> Expression:
        return self._new_binary_comparison(other, operator.gt)

    def __ge__(self, other) -> Expression:
        return self._new_binary_comparison(other, operator.ge)

    def __ne__(self, other) -> Expression:
        return self._new_binary_comparison(other, operator.ne)

    def __eq__(self, other) -> Expression:
        return self._new_binary_comparison(other, operator.eq)

    def __call__(self, *args, **kwargs) -> Expression:
        return self.new_function_application(
            self,
            [
                arg if isinstance(arg, Expression) else self.new_constant(arg, None)
                for arg in args
            ],
        )

    def __bool__(self):
        """
        We've overloaded == to return symbolic expressions (applications of equality).
        However, the operator == is used in important functions (such as `__hash__`),
        so in those contexts it is important that it still provides the original boolean result.
        This is done by providing this Expression -> bool converter,
        which evaluates expressions representing constant booleans and equalities.
        Refer to test/quick_tests/symbolic/z3_usage_test.py:test_z3_eq_bool() for how Z3 deals with a similar problem.
        """
        # TODO: why not convert applications of other operators such as !=, <, <= etc?
        match self:
            case Constant(value=value) if isinstance(value, bool):
                return value
            case FunctionApplication(
                function=Constant(value=operator.eq), arguments=[lhs, rhs]
            ):
                return lhs.syntactic_eq(rhs)
            case _:
                raise NotImplementedError("Expression cannot be converted to Boolean")


class AtomicExpression(Expression, ABC):
    @property
    @abstractmethod
    def atom(self) -> Any:
        pass

    @property
    def subexpressions(self) -> List[Expression]:
        return []

    def replace(
            self, from_expression: Expression, to_expression: Expression
    ) -> Expression:
        if from_expression.syntactic_eq(self):
            return to_expression
        else:
            return self

    def set(self, i: int, new_expression: Expression) -> Expression:
        raise IndexError(f"{type(self)} has no subexpressions, so you cannot set().")


class Constant(AtomicExpression, ABC):
    @property
    def form_kind(self) -> type[Expression]:
        return Constant

    @property
    def value(self) -> Any:
        return self.atom

    def __str__(self) -> str:
        return f"{self.value}"


class Variable(AtomicExpression, ABC):
    @property
    def form_kind(self) -> type[Expression]:
        return Variable

    @property
    def name(self) -> str:
        return self.atom

    def __str__(self) -> str:
        return f"{self.name}"


class FunctionApplication(Expression, ABC):
    __match_args__ = ("function", "arguments", "number_of_arguments")

    @property
    def form_kind(self) -> type[Expression]:
        return FunctionApplication

    @property
    @abstractmethod
    def function(self) -> Expression:
        pass

    @property
    @abstractmethod
    def number_of_arguments(self) -> int:
        # in some implementation getting number_of_arguments without calling arguments is useful,
        # e.g., a lazy implementation where arguments are only evaluated when used
        # Note: this is not `arity`, which is a property of a function, not of a function application;
        # it is possible to have a partial list of arguments.
        pass

    @property
    @abstractmethod
    def arguments(self) -> List[Expression]:
        pass

    @property
    @abstractmethod
    def subexpressions(self) -> List[Expression]:
        pass

    def set(self, i: int, new_expression: Expression) -> Expression:
        if i == 0:
            return self.new_function_application(new_expression, self.arguments)
        elif (argument_index := i - 1) < self.number_of_arguments:
            arguments = list(self.arguments)
            arguments[argument_index] = new_expression
            return self.new_function_application(self.function, arguments)
        else:
            raise IndexError(
                f"Out of scope. Function application only has {self.number_of_arguments} arguments "
                f"but you are setting {i - 1}th arguments."
            )

    def replace(
            self, from_expression: Expression, to_expression: Expression
    ) -> Expression:
        if from_expression.syntactic_eq(self):
            return to_expression

        # recursively do the replacement
        new_subexpressions = [
            e.replace(from_expression, to_expression) for e in self.subexpressions
        ]
        return self.new_function_application(
            new_subexpressions[0], new_subexpressions[1:]
        )

    def __str__(self) -> str:
        argument_str = ",".join([str(arg) for arg in self.arguments])
        return f"{self.function}({argument_str})"


class Context(Expression, ABC):
    @property
    @abstractmethod
    def dict(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def unsatisfiable(self) -> bool:
        pass

    @property
    @abstractmethod
    def satisfiability_is_known(self) -> bool:
        pass

    @property
    def and_priority(self) -> int:
        """So that conjoining anything with a context object `c` causes c.__and__ to be called."""
        return 1

    def _is_known_to_imply_fastpath(self, expression: Expression) -> Optional[bool]:
        """
        Provides subclasses with a way to provide an implication decision procedure in some faster way
        than the full-fledged methods.
        This is used by Z3ContextExpressionDummy which does not remember any accumulated constraints ("known nothing")
        and only checks if argument `expression` is true or false.
        That class is used in an experiment in which we know in advance no unsatisfiable conditions will happen,
        so we might as well not even check.
        TODO: check what happens in Z3ContextExpressionDummy when object is not true or false.
        It seems the slow decision procedure
        will be unnecessarily followed (unnecessary because Z3ContextExpressionDummy "knows nothing" anyway).
        TODO: either remove this as a hack, or make it better documented and supported.
        """
        return None

    def is_known_to_imply(self, expression: Expression) -> bool:
        """
        context implies expression iff (context => expression) is valid;
        which means not (context => expression) is unsatisfiable;
        which means not (not context or expression) is unsatisfiable;
        which means context and not expression is unsatisfiable.
        """
        if (fast_result := self._is_known_to_imply_fastpath(expression)) is not None:
            return fast_result

        new_context = self & ~expression
        if new_context.satisfiability_is_known:
            return new_context.unsatisfiable
        else:
            return False


class AbelianOperation(Constant, ABC):
    """
    An Abelian operation is a commutative, associative binary operation with an identity element:
    https://en.wikipedia.org/wiki/Abelian_group
    """

    @property
    @abstractmethod
    def identity(self) -> Expression:
        """
        An identity element on a set is an element of the set which leaves unchanged every element of the set
        when the operation is applied.
        E.g., for the operation "add" on natural numbers, 0 is the identity element. Because for all x, x + 0 = x.
        """
        pass

    @property
    def element_type(self) -> ExpressionType:
        return self.identity.type

    @abstractmethod
    def inverse(self, expr: Expression) -> Expression:
        """
        The inverse `i` of any element `a` in an Abelian group is defined by:
            i OP a = identity
        where OP is the Abelian operation.
        E.g., for integer addition,
            (-a) + a = 0
        so the inverse of element `a` is `-a`.
        """
        pass


class QuantifierExpression(Expression, ABC):
    """
    A more appropriate (but too long) name would be GeneralizedQuantifierExpression.
    Here the word "quantifier" does not only refer to logical quantifiers as in "forall" or "exists",
    but rather, the expression of an
    iterated binary operation (https://en.wikipedia.org/wiki/Iterated_binary_operation).

    This, in a sense, is a *generalized* version of quantifier: the operation of "forall" is `and`, the operation
    of "exists" is `or`, the operation of "Sigma" is `sum`, and "Pi" `multiplication`.
    """

    def __init__(self, expression_type: ExpressionType):
        super().__init__(expression_type)
        self._subexpressions = None

    @property
    def form_kind(self) -> type[Expression]:
        return QuantifierExpression

    @property
    @abstractmethod
    def operation(self) -> AbelianOperation:
        """
        operation is a Constant wrapping a function.
        E.g., integer Summation's operation is Constant(operator.add, Callable[[int,int],int]).
        """
        pass

    @property
    @abstractmethod
    def index(self) -> Variable:
        pass

    @property
    @abstractmethod
    def constraint(self) -> Context:
        """User can expect the returning `expression` to be a Boolean Expression, i.e., expression.type == bool."""
        pass

    @property
    @abstractmethod
    def body(self) -> Expression:
        pass

    @property
    def subexpressions(self) -> List[Expression]:
        if self._subexpressions is None:
            self._subexpressions = [self.operation, self.index, self.constraint, self.body]
        return self._subexpressions

    @property
    @abstractmethod
    def is_integral(self) -> bool:
        """
        @return: whether this QuantifierExpression is an "Integration" instead of a "Summation"
        Currently only makes sense when self.operation.value == operator.add
        This is a quick hack to support integral.
        TODO: clean up.
        """
        pass

    def set(self, i: int, new_expression: Expression) -> QuantifierExpression:
        subexpressions = list(self.subexpressions)
        subexpressions[i] = new_expression
        return self.new_quantifier_expression(
            *subexpressions, is_integral=self.is_integral
        )

    def replace(
            self, from_expression: Expression, to_expression: Expression
    ) -> Expression:
        if from_expression.syntactic_eq(self):
            return to_expression

        # recursively do the replacement
        new_subexpressions = [
            e.replace(from_expression, to_expression) for e in self.subexpressions
        ]
        return self.new_quantifier_expression(
            *new_subexpressions, is_integral=self.is_integral
        )

    def set_operation(self, new_operation: Expression) -> QuantifierExpression:
        """
        set_*() are a series of functions wrapping set() for better readability.
        Like set(), it does not modify `self`, but instead returns a new expression.
        """
        return self.set(0, new_operation)

    def set_index(self, new_index: Expression) -> QuantifierExpression:
        return self.set(1, new_index)

    def set_constraint(self, new_constraint: Expression) -> QuantifierExpression:
        return self.set(2, new_constraint)

    def set_body(self, new_body: Expression) -> QuantifierExpression:
        return self.set(3, new_body)

    def __str__(self) -> str:
        sign = (
            "Integral"
            if self.is_integral and self.operation.value == operator.add
            else f"Q_{self.operation}"
        )
        return f"{sign}({self.index}:{self.constraint}, {self.body})"
