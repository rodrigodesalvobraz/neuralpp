from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import sympy
from sympy import Poly, collect
import operator
import fractions

import neuralpp.symbolic.functions as functions
from neuralpp.symbolic.basic_expression import basic_add_operation, BasicConstant
from neuralpp.symbolic.expression import (
    AbelianOperation,
    Constant,
    Context,
    Expression,
    FunctionApplication,
    QuantifierExpression,
    Variable,
)
from neuralpp.symbolic.parameters import global_parameters
from neuralpp.symbolic.profiler import Profiler
from neuralpp.util.callable_util import (
    ExpressionType,
    infer_sympy_function_type,
    infer_sympy_object_type,
    python_callable_to_sympy_function,
    sympy_function_to_python_callable,
    return_type_after_application,
)
from neuralpp.util.symbolic_error_util import (
    FunctionNotTypedError,
    NotTypedError,
    UnknownError,
)
from neuralpp.util.sympy_util import (
    fold_sympy_piecewise,
    is_sympy_uninterpreted_function,
    is_sympy_integral,
    is_sympy_value,
    is_sympy_sum,
    sympy_piecewise_to_if_then_else
)
from neuralpp.util.util import distinct_pairwise, update_consistent_dict


# In this file's doc, I try to avoid the term `sympy expression` because it could mean both sympy.Expr (or sympy.Basic)
# and SymPyExpression. I usually use "sympy object" to refer to the former and "expression" to refer to the latter.


def _build_type_dict(
    sympy_arguments: SymPyExpression, type_dict: Dict[sympy.Basic, ExpressionType]
) -> None:
    update_consistent_dict(type_dict, sympy_arguments.type_dict)


def _build_type_dict_from_sympy_arguments(
    sympy_arguments: List[SymPyExpression],
) -> Dict[sympy.Basic, ExpressionType]:
    """
    Assumption: each element in sympy_arguments has a proper type_dict.
    Returns: a proper type_dict with these arguments joint
    """
    result = {}
    for sympy_argument in sympy_arguments:
        _build_type_dict(sympy_argument, result)
    return result


class SymPyExpression(Expression, ABC):
    def __init__(
        self,
        sympy_object: sympy.Basic,
        expression_type: ExpressionType,
        type_dict: Dict[sympy.Basic, ExpressionType],
    ):
        if expression_type is None:
            raise NotTypedError
        super().__init__(expression_type)
        self._sympy_object = sympy_object
        self._type_dict = type_dict

    @property
    def sympy_object(self):
        return self._sympy_object

    @property
    def type_dict(self) -> Dict[sympy.Basic, ExpressionType]:
        return self._type_dict

    def internal_object_eq(self, other) -> bool:
        match other:
            case SymPyExpression(
                sympy_object=other_sympy_object, type_dict=other_type_dict
            ):
                return (
                    self.sympy_object == other_sympy_object
                    and self.type_dict == other_type_dict
                )
            case _:
                return False

    @staticmethod
    def collect(expression: Expression, index: Variable) -> Expression:
        assert isinstance(expression, SymPyExpression)
        assert isinstance(index, SymPyExpression)
        new_sympy_object = collect(expression.sympy_object, index.sympy_object)
        return SymPyExpression.from_sympy_object(new_sympy_object, expression.type_dict)

    @staticmethod
    def symbolic_sum(
        body: Expression,
        index: Variable,
        lower_bound: Expression,
        upper_bound: Expression,
    ) -> Optional[Expression]:
        """try to compute the sum symbolically, if fails, return None"""
        try:
            body, index, lower_bound, upper_bound = [
                SymPyExpression._convert(argument)
                for argument in [body, index, lower_bound, upper_bound]
            ]
            type_dict = _build_type_dict_from_sympy_arguments(
                [body, index, lower_bound, upper_bound]
            )
            return SymPyExpression.from_sympy_object(
                sympy.Sum(
                    body.sympy_object,
                    (
                        index.sympy_object,
                        lower_bound.sympy_object,
                        upper_bound.sympy_object,
                    ),
                ).doit(),
                type_dict,
            )
        except Exception as exc:
            return None

    @staticmethod
    def symbolic_integral(
        body: Expression,
        index: Variable,
        lower_bound: Expression,
        upper_bound: Expression,
        profiler: Profiler,
    ) -> Optional[Expression]:
        """try to compute the integral symbolically, if fails, return None. Cached version."""
        try:
            with profiler.profile_section("convert"):
                body, index, lower_bound, upper_bound = [SymPyExpression._convert(argument)
                                                         for argument in [body, index, lower_bound, upper_bound]]
                type_dict = _build_type_dict_from_sympy_arguments([body, index, lower_bound, upper_bound])
            if body.sympy_object.is_Poly and index.sympy_object in body.sympy_object.gens:
                body_poly = body.sympy_object
            else:
                with profiler.profile_section("to poly"):
                    body_poly = Poly(body.sympy_object, index.sympy_object)
            with profiler.profile_section("poly integrate"):
                big_f = body_poly.integrate()
            with profiler.profile_section("substitute"):
                # print(f"replace {big_f.replace(index.sympy_object, upper_bound.sympy_object)}")
                assert isinstance(big_f, Poly)
                b = big_f.replace(index.sympy_object, upper_bound.sympy_object)
                a = big_f.replace(index.sympy_object, lower_bound.sympy_object)
            with profiler.profile_section("compute diff"):
                diff = b - a
            # if diff.has(index.sympy_object):
            #     raise ValueError(f"{diff}\nhas {index.sympy_object}???")
            with profiler.profile_section("wrap"):
                result = SymPyExpression.from_sympy_object(diff, type_dict)
            return result

        except Exception as exc:
            raise exc

    @classmethod
    def new_constant(
        cls, value: Any, type_: Optional[ExpressionType] = None
    ) -> SymPyConstant:
        # if a string contains a whitespace it'll be treated as multiple variables in sympy.symbols
        if isinstance(value, sympy.Basic):
            sympy_object = value
        elif isinstance(value, bool):
            sympy_object = sympy.S.true if value else sympy.S.false
        elif isinstance(value, int):
            sympy_object = sympy.Integer(value)
        elif isinstance(value, float):
            sympy_object = sympy.Float(value)
        elif isinstance(value, fractions.Fraction):
            sympy_object = sympy.Rational(value)
        elif isinstance(value, str):
            sympy_object = sympy.core.function.UndefinedFunction(value)
            if type_ is None:
                raise FunctionNotTypedError
        else:
            try:
                sympy_object = python_callable_to_sympy_function(value)
            except Exception as exc:
                raise ValueError(
                    f"SymPyConstant does not support {type(value)}: "
                    f"unable to turn into a sympy representation internally"
                ) from exc
        return SymPyConstant(sympy_object, type_)

    @classmethod
    def new_variable(cls, name: str, type_: ExpressionType) -> SymPyVariable:
        # if a string contains a whitespace it'll be treated as multiple variables in sympy.symbols
        if " " in name:
            raise ValueError(f"`{name}` should not contain a whitespace!")
        sympy_var = sympy.symbols(name)
        return SymPyVariable(sympy_var, type_)

    @classmethod
    def new_function_application(
        cls, function: Expression, arguments: List[Expression]
    ) -> SymPyExpression:
        # we cannot be lazy here because the goal is to create a sympy object, so arguments must be
        # recursively converted to sympy object
        match function:
            # first check if function is of SymPyConstant, where sympy_function is assumed to be a sympy function,
            # and we don't need to convert it.
            case SymPyConstant(sympy_object=sympy_function):
                return (
                    SymPyFunctionApplication.from_sympy_function_and_general_arguments(
                        sympy_function, arguments
                    )
                )
            # if function is not of SymPyConstant but of Constant, then it is assumed to be a python callable
            case Constant(value=python_callable):
                # during the call, ValueError will be implicitly raised if we cannot convert
                if python_callable == functions.conditional:
                    # special case of Conditional, this allows us to differentiate
                    # SymPyConditionalFunctionApplication (for if-then-else) and SymPyPiecewise (for piecewise)
                    if_, then, else_ = [
                        SymPyExpression._convert(argument) for argument in arguments
                    ]
                    type_dict = _build_type_dict_from_sympy_arguments(
                        [if_, then, else_]
                    )
                    piecewise = sympy.Piecewise(
                        (then.sympy_object, if_.sympy_object),
                        (else_.sympy_object, True),
                    )
                    return SymPyConditionalFunctionApplication(piecewise, type_dict)
                sympy_function = python_callable_to_sympy_function(python_callable)
                return (
                    SymPyFunctionApplication.from_sympy_function_and_general_arguments(
                        sympy_function, arguments
                    )
                )
            case Variable(name=name, type=type_):
                sympy_function = sympy.Function(name)
                return (
                    SymPyFunctionApplication.from_sympy_function_and_general_arguments(
                        sympy_function, arguments, type_
                    )
                )
            case FunctionApplication(_, _):
                raise ValueError("The function must be a python callable.")
            case _:
                raise ValueError("Unknown case.")

    @classmethod
    def new_quantifier_expression(
        cls,
        operation: Constant,
        index: Variable,
        constraint: Expression,
        body: Expression,
        is_integral: bool,
    ) -> Expression:
        raise NotImplementedError()

    @classmethod
    def pythonize_value(cls, value: sympy.Basic) -> Any:
        if isinstance(value, sympy.Integer):
            return int(value)
        elif isinstance(value, sympy.Float):
            return float(value)
        elif isinstance(value, sympy.Rational):
            return fractions.Fraction(value)
        elif isinstance(value, sympy.logic.boolalg.BooleanAtom):
            return bool(value)
        elif isinstance(value, sympy.core.function.UndefinedFunction):
            return str(value)  # uninterpreted function
        else:
            try:
                return sympy_function_to_python_callable(value)
            except Exception as exc:
                raise ValueError(f"Cannot pythonize {value}.") from exc

    @staticmethod
    def from_sympy_object(
        sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]
    ) -> SymPyExpression:
        # Here we just try to find a type of expression for sympy object.
        if isinstance(sympy_object, sympy.Symbol):
            return SymPyVariable(sympy_object, type_dict[sympy_object])
        elif is_sympy_value(sympy_object):
            return SymPyConstant(
                sympy_object, infer_sympy_object_type(sympy_object, type_dict)
            )
        elif is_sympy_sum(sympy_object):
            return SymPySummation(sympy_object, type_dict)
        elif is_sympy_integral(sympy_object):
            raise NotImplementedError("expect sympy to eliminate integral sign. TODO")
        else:
            return SymPyFunctionApplication(sympy_object, type_dict)

    @classmethod
    def convert(cls, from_expression: Expression) -> SymPyExpression:
        return cls._convert(from_expression)

    def __hash__(self):
        return self.sympy_object.__hash__()


class SymPyVariable(SymPyExpression, Variable):
    def __init__(self, sympy_object: sympy.Basic, expression_type: ExpressionType):
        SymPyExpression.__init__(
            self, sympy_object, expression_type, {sympy_object: expression_type}
        )

    @property
    def atom(self) -> str:
        return str(self._sympy_object)


class SymPyConstant(SymPyExpression, Constant):
    def __init__(
        self,
        sympy_object: sympy.Basic,
        expression_type: Optional[ExpressionType] = None,
    ):
        if expression_type is None:
            expression_type = infer_sympy_object_type(sympy_object, {})
        SymPyExpression.__init__(
            self, sympy_object, expression_type, {}
        )  # type_dict only records variables

    @property
    def atom(self) -> Any:
        return SymPyExpression.pythonize_value(self._sympy_object)


class SymPyFunctionApplicationInterface(SymPyExpression, FunctionApplication, ABC):
    def replace(self, from_expression: Expression, to_expression: Expression) -> Expression:
        with sympy.evaluate(global_parameters.sympy_evaluate):
            # a fast path
            from_expression_sympy = SymPyExpression.convert(from_expression)
            to_expression_sympy = SymPyExpression.convert(to_expression)
            return SymPyExpression.from_sympy_object(self.sympy_object.replace(from_expression_sympy.sympy_object, to_expression_sympy.sympy_object), _build_type_dict_from_sympy_arguments([self, to_expression_sympy]))

    @property
    def function(self) -> Expression:
        if self._sympy_object.func == Poly:
            return BasicConstant(functions.identity, Callable[[float], float])

        if is_sympy_uninterpreted_function(self._sympy_object.func):
            return SymPyVariable(self._sympy_object.func, self.function_type)
        else:
            return SymPyConstant(self._sympy_object.func, self.function_type)

    @property
    @abstractmethod
    def function_type(self) -> ExpressionType:
        pass

    @property
    def arguments(self) -> List[Expression]:
        if self._sympy_object.func == Poly:
            native_arguments = [self.native_arguments[0]]
        else:
            native_arguments = self.native_arguments
        return [
            SymPyExpression.from_sympy_object(argument, self.type_dict)
            for argument in native_arguments
        ]

    @property
    @abstractmethod
    def native_arguments(self) -> Tuple[sympy.Basic, ...]:
        pass

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.function] + self.arguments


class SymPyFunctionApplication(SymPyFunctionApplicationInterface):
    def __init__(
        self, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]
    ):
        """
        Calling by function_type=None asks this function to try to infer the function type.
        If the caller knows the function_type, it should always set function_type to a non-None value.
        This function always set type_dict[sympy_object] with the new (inferred or supplied) function_type value.
        The old value, if exists, is only used for consistency checking.
        """
        if sympy_object.is_Poly:
        # if True:
            self._function_type = Callable[[], float]
            SymPyExpression.__init__(self, sympy_object, float, type_dict)
            return

        if not sympy_object.args:
            raise TypeError(f"not a function application. {sympy_object}")

        if sympy_object.func in type_dict:
            # this happens iff sympy_object is an uninterpreted function, whose type cannot be inferred
            self._function_type = type_dict[sympy_object.func]
        else:
            self._function_type = infer_sympy_function_type(sympy_object, type_dict)
        return_type = return_type_after_application(
            self._function_type, len(sympy_object.args)
        )
        SymPyExpression.__init__(self, sympy_object, return_type, type_dict)

    @property
    def is_polynomials(self) -> bool:
        return self.sympy_object.is_Poly

    @property
    def function_type(self) -> ExpressionType:
        return self._function_type

    @property
    def number_of_arguments(self) -> int:
        return len(self.native_arguments)

    @property
    def native_arguments(self) -> Tuple[sympy.Basic, ...]:
        """faster than arguments"""
        return self._sympy_object.args  # sympy f.args returns a tuple

    @staticmethod
    def from_sympy_function_and_general_arguments(
        sympy_function: sympy.Basic,
        arguments: List[Expression],
        uninterpreted_function_type: ExpressionType = None,
    ) -> SymPyExpression:
        sympy_arguments = [SymPyExpression._convert(argument) for argument in arguments]
        type_dict = _build_type_dict_from_sympy_arguments(sympy_arguments)

        if is_sympy_uninterpreted_function(sympy_function):
            # If the function is uninterpreted, we must also save its type information in type_dict,
            # since its type cannot be inferred.
            if uninterpreted_function_type is None:
                raise ValueError(
                    f"uninterpreted function {sympy_function} has no type!"
                )
            type_dict[sympy_function] = uninterpreted_function_type

        if sympy_function == sympy.Min or sympy_function == sympy.Max:
            # see test/sympy_test.py: test_sympy_bug()
            sympy_object = sympy_function(
                *[sympy_argument.sympy_object for sympy_argument in sympy_arguments],
                evaluate=global_parameters.sympy_evaluate,
            )
        else:
            with sympy.evaluate(global_parameters.sympy_evaluate):
                # If we want to preserve the symbolic structure, we need to stop evaluation by setting
                # global_parameters.sympy_evaluate to False (or Add(1,1) will be 2 in sympy).
                native_arguments = [
                    sympy_argument.sympy_object for sympy_argument in sympy_arguments
                ]
                if sympy_function == sympy.Piecewise:
                    sympy_object = sympy_function(
                        *distinct_pairwise(native_arguments)
                    )  # distinct_pairwise() turns [a,b,c,d,..] into [(a,b),(c,d),..]
                else:
                    if any(arg.is_Poly for arg in native_arguments):
                        if sympy_function == sympy.Mul:
                            if len(native_arguments) != 2:
                                [print(arg) for arg in native_arguments]
                                raise
                            if native_arguments[0].is_Poly and native_arguments[1].is_Poly:
                                sympy_object = native_arguments[0].mul(native_arguments[1])
                                assert sympy_object.is_Poly
                            elif native_arguments[0].is_Poly:
                                # try:
                                sympy_object = native_arguments[0].mul(native_arguments[1])
                                assert sympy_object.is_Poly
                                # except Exception as _:
                                #     sympy_object = native_arguments[0].as_expr() * native_arguments[1]
                            else:  # native_arguments[1].is_Poly
                                # try:
                                sympy_object = native_arguments[1].mul(native_arguments[0])
                                # except Exception as _:
                                #     sympy_object = native_arguments[1].as_expr() * native_arguments[0]
                        elif sympy_function == sympy.Add:
                            sympy_object = sympy.Add(*native_arguments, evaluate=False)
                            # assert native_arguments[0].is_Poly
                            # sympy_object = native_arguments[0]
                            # for arg in native_arguments[1:]:
                            #     sympy_object = sympy_object.add(arg)
                        else:
                            raise RuntimeError(f"Unknown function {sympy_function}")
                    else:
                        sympy_object = sympy_function(*native_arguments)

        if global_parameters.sympy_evaluate:
            # if sympy_evaluate is True, we don't necessarily return a FunctionApplication.
            # E.g., sympy_object = (a + y) - y would be a.
            return SymPyExpression.from_sympy_object(sympy_object, type_dict)
        else:
            return SymPyFunctionApplication(sympy_object, type_dict)

    def __new__(
        cls, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]
    ):
        if sympy_object.func == sympy.Piecewise:
            return SymPyPiecewise(sympy_object, type_dict)
        else:
            return super().__new__(cls)


class SymPyConditionalFunctionApplication(SymPyFunctionApplicationInterface):
    def __init__(
        self, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]
    ):
        if sympy_object.func != sympy.Piecewise:
            raise TypeError(
                f"Can only create conditional function application when function is sympy.Piecewise. "
                f"Current function is {sympy_object.func}"
            )
        if not sympy_object.args[-1][
            1
        ]:  # the clause condition must be True otherwise it's not an if-then-else
            raise TypeError("Missing else clause.")
        sympy_object = fold_sympy_piecewise(sympy_object.args)
        self._then_type = infer_sympy_object_type(sympy_object.args[0][0], type_dict)
        SymPyExpression.__init__(self, sympy_object, self._then_type, type_dict)

    @property
    def function(self) -> Expression:
        from neuralpp.symbolic.constants import (
            if_then_else_function,
        )  # have to stay here otherwise we have circular import?

        return if_then_else_function(
            self._then_type
        )  # should be treated as "if" instead of piecewise

    @property
    def function_type(self) -> ExpressionType:
        return Callable[[bool, self._then_type, self._then_type], self._then_type]

    @property
    def number_of_arguments(self) -> int:
        return 3

    @property
    def native_arguments(self) -> Tuple[sympy.Basic, ...]:
        return sympy_piecewise_to_if_then_else(self.sympy_object)


class SymPyPiecewise(SymPyFunctionApplicationInterface):
    def __init__(
        self, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]
    ):
        if sympy_object.func != sympy.Piecewise:
            raise TypeError(
                "Can only create piecewise function application when function is sympy.Piecewise."
            )
        self._then_type = infer_sympy_object_type(sympy_object.args[0][0], type_dict)
        SymPyExpression.__init__(self, sympy_object, self._then_type, type_dict)

    @property
    def function(self) -> Expression:
        return SymPyConstant(sympy.Piecewise, self.function_type)

    @property
    def function_type(self) -> ExpressionType:
        return Callable[[bool, self._then_type, self._then_type], self._then_type]

    @property
    def number_of_arguments(self) -> int:
        return len(self.sympy_object.args) * 2

    @property
    def arguments(self) -> List[Expression]:
        return [
            SymPyExpression.from_sympy_object(element, self.type_dict)
            for expr, cond in self.native_arguments
            for element in (expr, cond)
        ]

    @property
    def native_arguments(self) -> Tuple[sympy.Basic, ...]:
        return self.sympy_object.args


class SymPyContext(SymPyFunctionApplication, Context):
    """
    SymPyContext is just a SymPyFunctionApplication, which always raises when asked for satisfiability
    since we don't know.
    We create a dictionary from the function application at initialization."""

    def __init__(
        self, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]
    ):
        SymPyFunctionApplication.__init__(self, sympy_object, type_dict)
        (
            self._dict,
            self._unknown,
            self._unsatisfiable,
        ) = self._context_to_variable_value_dict(self)
        if self._unsatisfiable:
            # if unsat, looking up dict for value does not make sense, as any value can be detailed
            self._dict = {}

    @property
    def dict(self) -> Dict[str, Any]:
        """User should not write to the return value."""
        return self._dict

    @property
    def unsatisfiable(self) -> bool:
        if self._unknown:
            raise UnknownError()
        else:
            return self._unsatisfiable

    @property
    def satisfiability_is_known(self) -> bool:
        return not self._unknown

    def _context_to_variable_value_dict(
        self,
        context: FunctionApplication,
    ) -> Tuple[Dict[str, Any], bool, bool]:
        """
        Returns a dictionary, and two booleans: first indicating whether its satisfiability is unknown,
        second indicating whether it is unsatisfiable (if its satisfiability is known)
        """
        return self._context_to_variable_value_dict_helper(context, {})

    def _context_to_variable_value_dict_helper(
        self,
        context: FunctionApplication,
        variable_to_value: Dict[str, Any],
        unknown: bool = False,
        unsatisfiable: bool = False,
    ) -> Tuple[Dict[str, Any], bool, bool]:
        """
        variable_to_value: the mutable argument also serves as a return value.
        By default, we assume the context's satisfiability can be known and is True.
        If the context has multiple assignments (e.g., x==3 and x==5), the context is unsatisfiable.
        If the context is anything other than a conjunction of equalities, the context's satisfiability is unknown.
        """
        match context:
            case FunctionApplication(
                function=Constant(value=operator.and_), arguments=arguments
            ):
                # the conjunctive case
                for sub_context in arguments:
                    (
                        variable_to_value,
                        unknown,
                        unsatisfiable,
                    ) = self._context_to_variable_value_dict_helper(
                        sub_context, variable_to_value, unknown, unsatisfiable
                    )
            case FunctionApplication(
                function=Constant(value=operator.eq),
                arguments=[Variable(name=variable), Constant(value=value)],
            ) | FunctionApplication(
                function=Constant(value=operator.eq),
                arguments=[Constant(value=value), Variable(name=variable)],
            ):
                # the leaf case
                if (
                    variable in variable_to_value
                    and variable_to_value[variable] != value
                ):
                    unsatisfiable = True
                variable_to_value[variable] = value
            # all other cases makes the satisfiability unknown
            case _:
                unknown = True
        return variable_to_value, unknown, unsatisfiable


class SymPySummation(SymPyExpression, QuantifierExpression):
    def __init__(
        self, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]
    ):
        expression_type = infer_sympy_object_type(sympy_object.args[0], type_dict)
        # sympy.Sum(body, (index, lower, upper))
        super().__init__(sympy_object, expression_type, type_dict)

    @property
    def operation(self) -> AbelianOperation:
        return basic_add_operation(self.type)

    @property
    def index(self) -> SymPyVariable:
        variable = self.sympy_object.args[1][0]
        return SymPyVariable(variable, self.type_dict[variable])

    @property
    def constraint(self) -> Context:
        from .z3_expression import Z3SolverExpression

        empty_context = Z3SolverExpression()
        return (
            empty_context
            & (self.index >= self.lower_bound)
            & (self.index <= self.upper_bound)
        )

    @property
    def body(self) -> Expression:
        return SymPyExpression.from_sympy_object(
            self.sympy_object.args[0], self.type_dict
        )

    @property
    def lower_bound(self) -> SymPyExpression:
        lower = self.sympy_object.args[1][1]
        return SymPyExpression.from_sympy_object(lower, self.type_dict)

    @property
    def upper_bound(self) -> SymPyExpression:
        upper = self.sympy_object.args[1][2]
        return SymPyExpression.from_sympy_object(upper, self.type_dict)

    @property
    def is_integral(self) -> bool:
        return False


def make_piecewise(arguments: List[Expression]):
    from .basic_expression import BasicFunctionApplication
    return BasicFunctionApplication(SymPyConstant(sympy.Piecewise, Callable[[], float]), arguments)
