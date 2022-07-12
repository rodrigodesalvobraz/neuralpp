from __future__ import annotations  # to support forward reference for recursive type reference

import fractions
import typing
import operator
from typing import List, Any, Optional, Type, Callable, Dict
from abc import ABC, abstractmethod

# typing.Callable can be ambiguous.
# Consider the following two tests:
#   >>> isinstance(int, Type)
#   True
#   >>> isinstance(1, Type)
#   False
# So "int is an instance of Type" and "1 is not an instance of Type"
# But for Callable, we have
#   >>> isinstance(Callable[[int],int], Callable)
#   True
#   >>> isinstance(lambda x: x, Callable)
#   True
# Here the `Callable` in each case means differently, which makes `Callable` ambiguous.
#
# Imagine a typed language, we'd have
#   >>> isinstance(lambda (x: int) : int := x, int->int)
#   True
# and only one of the following two cases can be true:
# CASE ONE: `Callable` means "the type of all function types".
#   >>> isinstance(int->int, Callable)
#   True
#   >>> isinstance(lambda (x: int) : int := x, Callable)
#   False
# or
# CASE TWO: `Callable` means "the type of all functions"
#   >>> isinstance(int->int, Callable)
#   False
#   >>> isinstance(lambda (x: int) : int := x, Callable)
#   True
#
# It should be noted that our usage of `Callable` here means the first, i.e., "the type of all function types"
# And in our code we use `isinstance()` for cases like `isinstance(Callable[[int],int], Callable)`.
ExpressionType = Callable | Type


def return_type_after_application(callable_: Callable, number_of_arguments: int) -> ExpressionType:
    """ Given number_of_arguments (<=arity), return the return type after (partial) application. """
    argument_types, return_type = typing.get_args(callable_)
    arity = len(argument_types)
    if number_of_arguments > arity:
        raise ValueError(f"number_of_arguments {number_of_arguments} > arity {arity}.")
    elif number_of_arguments == arity:
        return return_type
    else:
        return Callable[argument_types[number_of_arguments:], return_type]


type_order_in_arithmetic = [fractions.Fraction, float, int]


def get_arithmetic_function_type_from_argument_types(argument_types: List[ExpressionType]) -> Callable:
    try:
        # e.g., if float + int, the return type is float
        return_type = type_order_in_arithmetic[min(map(type_order_in_arithmetic.index, argument_types))]
        return Callable[argument_types, return_type]
    except ValueError as err:
        raise ValueError(f"Can only infer the return type from arithmetic argument types: "
                         f"fractions.Fraction, float and int. {err}")


class Expression(ABC):
    def __init__(self, expression_type: ExpressionType):
        self._type = expression_type

    @property
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

        for sub_expr in self.subexpressions:
            if sub_expr.contains(target):
                return True

        return False

    @abstractmethod
    def syntactic_eq(self, other) -> bool:
        """
        Returns if self and other are syntactically equal, i.e.,
        that they are of subclass of Expression and their internal representation are equal.
        This method usually depends on subclass-specific library calls,
        e.g., Z3Expression.syntactic_eq() would leverage z3.eq().
        This method should be considered as a cheap way to check syntactic equality of two symbolic expressions.
        """
        pass

    def structure_eq(self, other) -> bool:
        """
        Returns if self and other are "structurally" equivalent, i.e., that they have the same Expression interfaces.
        E.g, a Z3Expression of "a + b" does not syntactic_eq() a SymPyExpression of "a + b", but a call of
        structure_eq() on the two should return True.
        This method is general and more expensive than syntactic_eq()
        """
        match self, other:
            case AtomicExpression(base_type=self_base_type, atom=self_atom, type=self_type),\
                    AtomicExpression(base_type=other_base_type, atom=other_atom, type=other_type):
                return self_base_type == other_base_type and self_type == other_type and self_atom == other_atom
            case FunctionApplication(subexpressions=self_subexpressions, type=self_type), \
                    FunctionApplication(subexpressions=other_subexpressions, type=other_type):
                return all(lhs.structure_eq(rhs) for lhs, rhs in zip(self_subexpressions, other_subexpressions)) and \
                       self_type == other_type
            case _:
                return False

    @classmethod
    @abstractmethod
    def new_constant(cls, value: Any, type_: Optional[ExpressionType]) -> Constant:
        """
        Value is expected to be a python object or a "native" object. E.g.,
        SymPyExpression.new_constant()'s legal input would have python `int` and `sympy.Integer`,
        but not `z3.Int`. Similarly, Z3Expression.new_constant()'s legal input has `int` and `z3.Int` but
        not `sympy.Integer`.
        """
        pass

    @classmethod
    @abstractmethod
    def new_variable(cls, name: str, type_: ExpressionType) -> Variable:
        pass

    @classmethod
    @abstractmethod
    def new_function_application(cls, function: Expression, arguments: List[Expression]) -> FunctionApplication:
        pass

    @classmethod
    def _convert(cls, from_expression: Expression) -> Expression:
        """ general helper for converting an Expression into this subclass of Expression. """
        if isinstance(from_expression, cls):
            return from_expression
        match from_expression:
            case Constant(value=value, type=type_):
                return cls.new_constant(value, type_)
            case Variable(name=name, type=type_):
                return cls.new_variable(name, type_)
            case FunctionApplication(function=function, arguments=arguments):
                return cls.new_function_application(function, arguments)
            case _:
                raise ValueError(f"invalid from_expression {from_expression}: {type(from_expression)}")

    @property
    def type(self) -> ExpressionType:
        return self._type

    @property
    def and_priority(self) -> int:
        """ This property is by default set to 0. Any subclass wishing to 'overshadow' in `and` operator may
        set this value higher. For example: if a and b are Expressions, both having __and__() overloaded:
        >>> a: Expression
        >>> b: Expression
        then
        >>> a & b
        would mean
        >>> a.__and__(b)
        However, if b is a subclass that overload `and_property` to a >0 value. Then
        >>> a & b
        would mean
        >>> b.__and__(a)

        In particualr, this is useful in `Context`, sicne we want `Context` object to always `overshadow` its neighbors.
        So that
        >>> literal & context
        would cause
        >>> context.__and__(literal)
        thus adding literal to the context (instead of creating a new expression where we lost the context information).
        """
        return 0

    def get_function_type(self) -> ExpressionType:
        """
        For example, a function (the declaration, not the application) can be:
            add_func_type = Callable[[int, int], int]
            add_func = Constant(operator.add, add_func_type)
        get_function_type() serves as a method to retrieve the "function type" of `add_func`, so we can expect:
            add_func.get_function_type() == add_func_type
        Note add_func can also be
            add_func = Variable("add", add_func_type)  # uninterpreted function
        or
            three_way_add = Variable("three_way_add", Callable[[int, int, int], int])
            add_func = FunctionApplication(three_way_add, Constant(0))
        In both cases we can expect
            add_func.get_function_type() == add_func_type
        """
        match self:
            case Constant(type=type_) | Variable(type=type_):
                return type_
            case FunctionApplication(function=function, number_of_arguments=num):
                return function.get_return_type(num)

    def get_return_type(self, number_of_arguments: int) -> ExpressionType:
        function_type = self.get_function_type()
        if not isinstance(function_type, Callable):
            raise TypeError(f"{self}'s function is not of function type.")
        return return_type_after_application(function_type, number_of_arguments)

    def _new_binary_arithmetic(self, other, operator_, function_type=None, reverse=False) -> Expression:
        return self._new_binary_operation(other, operator_, function_type, reverse, arithmetic=True)

    def _new_binary_boolean(self, other, operator_, reverse=False) -> Expression:
        return self._new_binary_operation(other, operator_, Callable[[bool, bool], bool], reverse, arithmetic=False)

    def _new_binary_comparison(self, other, operator_, function_type=None, reverse=False) -> Expression:
        return self._new_binary_operation(other, operator_, function_type, reverse, arithmetic=False)

    def _new_binary_operation(self, other, operator_, function_type=None, reverse=False, arithmetic=True) -> Expression:
        """
        Wrapper to make a binary operation in self's class. Tries to convert other to a Constant if it is not
        an Expression.
        E.g., if operator_ is `+`, other is `3`. return self + Constant(3).
        By default, self is the 1st argument and other is the 2nd.
        If reverse is set to True, it is reversed, so e.g., if operator_ is `-` and reverse is True, then return
        `other - self`.
        If `arithmetic` is True, the return type is inferred from the argument types. Otherwise, it's assumed to
        be bool.
        """
        if not isinstance(other, Expression):
            other = self.new_constant(other, None)  # we can only try to create constant, for variable we need type.
        arguments = [self, other] if not reverse else [other, self]
        if function_type is None:
            if arithmetic:
                function_type = get_arithmetic_function_type_from_argument_types([arguments[0].type, arguments[1].type])
            else:
                if arguments[0].type != arguments[1].type:
                    raise TypeError(f"Argument types mismatch: {arguments[0].type} != {arguments[1].type}.")
                function_type = Callable[[arguments[0].type, arguments[1].type], bool]
        return self.new_function_application(self.new_constant(operator_, function_type), arguments)

    def __add__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.add)

    # We can also write __radd__ = __add__. But it may be better to keep the order?
    def __radd__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.add, reverse=True)

    def __mul__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.mul)

    def __rmul__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.mul, reverse=True)

    def __truediv__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.truediv, reverse=True)

    def __sub__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.sub)

    def __rsub__(self, other: Any) -> Expression:
        return self._new_binary_arithmetic(other, operator.sub, reverse=True)

    def __neg__(self) -> Expression:
        return self.new_function_application(self.new_constant(operator.neg, Callable[[self.type], self.type]), [self])

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
        return self.new_function_application(self.new_constant(operator.invert, Callable[[bool], bool]), [self])

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
        return self.new_function_application(self,
                                             [arg if isinstance(arg, Expression) else self.new_constant(arg, None)
                                              for arg in args])


class AtomicExpression(Expression, ABC):
    @property
    @abstractmethod
    def base_type(self) -> str:
        pass

    @property
    @abstractmethod
    def atom(self) -> Any:
        pass

    @property
    def subexpressions(self) -> List[Expression]:
        return []

    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        if from_expression.syntactic_eq(self):
            return to_expression
        else:
            return self

    def set(self, i: int, new_expression: Expression) -> Expression:
        raise IndexError(f"{type(self)} has no subexpressions, so you cannot set().")

    def contains(self, target: Expression) -> bool:
        return self.syntactic_eq(target)


class Variable(AtomicExpression, ABC):
    @property
    def base_type(self) -> str:
        return "Variable"

    @property
    def name(self) -> str:
        return self.atom

    def __str__(self) -> str:
        return f'"{self.name}"'


class Constant(AtomicExpression, ABC):
    @property
    def base_type(self) -> str:
        return "Constant"

    @property
    def value(self) -> Any:
        return self.atom

    def __str__(self) -> str:
        return f"{self.value}"


class FunctionApplication(Expression, ABC):
    __match_args__ = ("function", "arguments", "number_of_arguments")

    @property
    @abstractmethod
    def function(self) -> Expression:
        pass

    @property
    @abstractmethod
    def number_of_arguments(self) -> int:
        # in some implementation getting number_of_arguments without calling arguments is useful,
        # e.g., a lazy implementation where arguments are only evaluated when used
        # Note: this is not `arity`, which is a property of a function, not a function application.
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

        # evaluate len after i != 0, if i == 0 we can be lazy
        if i - 1 < self.number_of_arguments:
            arguments = self.arguments
            arguments[i - 1] = new_expression
            return self.new_function_application(self.function, arguments)
        else:
            raise IndexError(f"Out of scope. Function application only has {self.number_of_arguments} arguments "
                             f"but you are setting {i - 1}th arguments.")

    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        if from_expression.syntactic_eq(self):
            return to_expression

        # recursively do the replacement
        new_subexpressions = [
            to_expression if e.syntactic_eq(from_expression) else e.replace(from_expression, to_expression)
            for e in self.subexpressions
        ]
        return self.new_function_application(new_subexpressions[0], new_subexpressions[1:])

    def __str__(self) -> str:
        argument_str = ",".join([str(arg) for arg in self.arguments])
        return f"{self.function}({argument_str})"


class Context(Expression, ABC):
    class UnknownError(ValueError, RuntimeError):
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
        """ So that conjoining anything with a context object `c` causes c.__and__ to be called. """
        return 1

    @property
    @abstractmethod
    def dict(self) -> Dict[str, Any]:
        pass


class NotTypedError(ValueError, TypeError):
    pass


class VariableNotTypedError(NotTypedError):
    pass


class FunctionNotTypedError(NotTypedError):
    pass
