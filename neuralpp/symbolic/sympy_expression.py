from __future__ import annotations

import typing
from typing import List, Any, Type, Callable, Tuple, Optional, Dict
from abc import ABC, abstractmethod

import sympy
from sympy import abc
import operator
import builtins
import fractions

from functools import cached_property
from neuralpp.symbolic.expression import Expression, FunctionApplication, Variable, Constant, \
    FunctionNotTypedError, NotTypedError, return_type_after_application, ExpressionType, Context, QuantifierExpression, \
    AbelianOperation
from neuralpp.symbolic.basic_expression import infer_python_callable_type, basic_add_operation
from neuralpp.util.util import update_consistent_dict
from neuralpp.symbolic.parameters import global_parameters
from neuralpp.symbolic.basic_expression import BasicSummation
import neuralpp.symbolic.functions as functions
from neuralpp.util.sympy_util import is_sympy_uninterpreted_function
from functools import cache


# In this file's doc, I try to avoid the term `sympy expression` because it could mean both sympy.Expr (or sympy.Basic)
# and SymPyExpression. I usually use "sympy object" to refer to the former and "expression" to refer to the latter.


# @cache
# def _get_sympy_integral(sympy_expression, x):
#     return sympy.Integral(sympy_expression, x).doit()
_cache = {}
def _get_sympy_integral(sympy_expression, x):
    key = (sympy_expression, x)
    if key in _cache:
        return _cache[key]
    else:
        _cache[key] = sympy.Integral(sympy_expression, x).doit()
        return -_cache[key]


def _infer_sympy_object_type(sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]) -> ExpressionType:
    """
    type_dict can be, for example, {a: int, b: int, c: float, f: int->int}.
    """
    match sympy_object:
        case sympy.Integer():
            return int
        case sympy.Float():
            return float
        case sympy.Rational():
            return fractions.Fraction
        case sympy.logic.boolalg.BooleanAtom():
            return bool
        case sympy.Piecewise():  # piecewise is not mapped in python_callable_and_sympy_function_relation
            return _infer_sympy_object_type(sympy_object.args[0][0], type_dict)
        case _:
            # We can look up type_dict for variable like `symbol("x")`.
            # A variable can also be an uninterpreted function `sympy.core.function.UndefinedFunction('f')`
            try:
                return type_dict[sympy_object]
            except KeyError:  # if it's not in type_dict, try figure out ourselves
                # Here, sympy_object must be function applications like 'x+y'
                if len(sympy_object.args) == 0:  # len(sympy_object.args) could raise (e.g, len(sympy.Add.args))
                    raise TypeError("expect function application")
                _, return_type = typing.get_args(_infer_sympy_function_type(sympy_object, type_dict))
                return return_type


def _infer_sympy_function_type(sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]) -> Callable:
    """
    Assume sympy_object is a function application, return the function type of the function application.
    Note this is different from infer_sympy_object_type, which returns the type of function application.
    E.g.,
    >>> from sympy.abc import a, b
    >>> _infer_sympy_function_type(a+b, {a:int, b:int})
    Callable[[int, int], int]
    >>> _infer_sympy_object_type(a+b, {a:int, b:int})
    int
    """
    return infer_python_callable_type(_sympy_function_to_python_callable(sympy_object.func),
                                      [_infer_sympy_object_type(arg, type_dict) for arg in sympy_object.args])


sympy_Sub = sympy.Lambda((abc.x, abc.y), abc.x - abc.y)
sympy_Neg = sympy.Lambda((abc.x,), -abc.x)  # "lambda x: (-1)*x"
sympy_Cond = sympy.Lambda((abc.i, abc.t, abc.e), sympy.Piecewise((abc.t, abc.i), (abc.e, True)))
sympy_Div = sympy.Lambda((abc.x, abc.y), abc.x / abc.y)
# Refer to sympy_simplification_test:test_unevaluate() for this design that uses sympy.Lambda()
python_callable_and_sympy_function_relation = [
    # boolean operation
    (operator.and_, sympy.And),
    (operator.or_, sympy.Or),
    (operator.invert, sympy.Not),
    (operator.xor, sympy.Xor),
    # comparison
    (operator.le, sympy.Le),
    (operator.lt, sympy.Lt),
    (operator.ge, sympy.Ge),
    (operator.gt, sympy.Gt),
    (operator.eq, sympy.Eq),
    (operator.ne, sympy.Ne),
    # arithmetic
    (operator.add, sympy.Add),
    (operator.mul, sympy.Mul),
    (operator.pow, sympy.Pow),
    (operator.sub, sympy_Sub),
    (operator.neg, sympy_Neg),
    (operator.truediv, sympy_Div),
    # min/max
    (builtins.min, sympy.Min),
    (builtins.max, sympy.Max),
    # conditional
    (functions.conditional, sympy_Cond),
    # identity
    (functions.identity, sympy.Id)
]
sympy_function_python_callable_dict = \
    {sympy_function: python_callable
     for python_callable, sympy_function in python_callable_and_sympy_function_relation}
python_callable_sympy_function_dict = \
    {python_callable: sympy_function
     for python_callable, sympy_function in python_callable_and_sympy_function_relation}


def _sympy_function_to_python_callable(sympy_function: sympy.Basic) -> Callable:
    try:
        return sympy_function_python_callable_dict[sympy_function]
    except KeyError:
        if sympy_function == sympy.Piecewise:
            return functions.conditional
        raise ValueError(f"SymPy function {sympy_function} is not recognized.")


def _python_callable_to_sympy_function(python_callable: Callable) -> sympy.Basic:
    try:
        return python_callable_sympy_function_dict[python_callable]
    except KeyError:
        raise ValueError(f"Python callable {python_callable} is not recognized.")


def _is_sympy_value(sympy_object: sympy.Basic) -> bool:
    return isinstance(sympy_object, sympy.Number) or \
           isinstance(sympy_object, sympy.logic.boolalg.BooleanAtom)


def _is_sympy_sum(sympy_object: sympy.Basic) -> bool:
    return isinstance(sympy_object, sympy.Sum)


def _is_sympy_integral(sympy_object: sympy.Basic) -> bool:
    return isinstance(sympy_object, sympy.Integral)


def _build_type_dict(sympy_arguments: SymPyExpression, type_dict: Dict[sympy.Basic, ExpressionType]) -> None:
    update_consistent_dict(type_dict, sympy_arguments.type_dict)


def _build_type_dict_from_sympy_arguments(sympy_arguments: List[SymPyExpression]) -> Dict[sympy.Basic, ExpressionType]:
    """
    Assumption: each element in sympy_arguments has a proper type_dict.
    Returns: a proper type_dict with these arguments joint
    """
    result = {}
    for sympy_argument in sympy_arguments:
        _build_type_dict(sympy_argument, result)
    return result


class SymPyExpression(Expression, ABC):
    def __init__(self, sympy_object: sympy.Basic, expression_type: ExpressionType,
                 type_dict: Dict[sympy.Basic, ExpressionType]):
        if expression_type is None:
            raise NotTypedError
        super().__init__(expression_type)
        self._sympy_object = sympy_object
        self._type_dict = type_dict

    @staticmethod
    def symbolic_sum(body: Expression, index: Variable, lower_bound: Expression, upper_bound: Expression) \
            -> Optional[Expression]:
        """ try to compute the sum symbolically, if fails, return None"""
        try:
            body, index, lower_bound, upper_bound = [SymPyExpression._convert(argument)
                                                     for argument in [body, index, lower_bound, upper_bound]]
            type_dict = _build_type_dict_from_sympy_arguments([body, index, lower_bound, upper_bound])
            return SymPyExpression.from_sympy_object(sympy.Sum(body.sympy_object,
                                                               (index.sympy_object,
                                                                lower_bound.sympy_object,
                                                                upper_bound.sympy_object,
                                                                ),
                                                               ).doit(),
                                                     type_dict)
        except Exception as exc:
            return None

    @staticmethod
    def symbolic_integral(body: Expression, index: Variable, lower_bound: Expression, upper_bound: Expression) \
            -> Optional[Expression]:
        """ try to compute the integral symbolically, if fails, return None"""
        try:
            body, index, lower_bound, upper_bound = [SymPyExpression._convert(argument)
                                                     for argument in [body, index, lower_bound, upper_bound]]
            print(f"body:{body.sympy_object}")
            type_dict = _build_type_dict_from_sympy_arguments([body, index, lower_bound, upper_bound])
            return SymPyExpression.from_sympy_object(sympy.Integral(body.sympy_object,
                                                                    (index.sympy_object,
                                                                     lower_bound.sympy_object,
                                                                     upper_bound.sympy_object,
                                                                     ),
                                                                    ).doit(),
                                                     type_dict)
        except Exception as exc:
            return None

    @staticmethod
    def symbolic_integral_cached(body: Expression, index: Variable, lower_bound: Expression, upper_bound: Expression) \
            -> Optional[Expression]:
        """ try to compute the integral symbolically, if fails, return None"""
        try:
            body, index, lower_bound, upper_bound = [SymPyExpression._convert(argument)
                                                     for argument in [body, index, lower_bound, upper_bound]]
            print(f"body:{body.sympy_object}")
            type_dict = _build_type_dict_from_sympy_arguments([body, index, lower_bound, upper_bound])
            indefinite_integral = _get_sympy_integral(body.sympy_object, index.sympy_object)
            difference = indefinite_integral.subs(index.sympy_object, upper_bound.sympy_object) - indefinite_integral.subs(index.sympy_object, lower_bound.sympy_object)
            return SymPyExpression.from_sympy_object(difference, type_dict)
        except Exception as exc:
            return None

    @classmethod
    def new_constant(cls, value: Any, type_: Optional[ExpressionType] = None) -> SymPyConstant:
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
                sympy_object = _python_callable_to_sympy_function(value)
            except Exception as exc:
                raise ValueError(f"SymPyConstant does not support {type(value)}: "
                                 f"unable to turn into a sympy representation internally") from exc
        return SymPyConstant(sympy_object, type_)

    @classmethod
    def new_variable(cls, name: str, type_: ExpressionType) -> SymPyVariable:
        # if a string contains a whitespace it'll be treated as multiple variables in sympy.symbols
        if ' ' in name:
            raise ValueError(f"`{name}` should not contain a whitespace!")
        sympy_var = sympy.symbols(name)
        return SymPyVariable(sympy_var, type_)

    @classmethod
    def new_function_application(cls, function: Expression, arguments: List[Expression]) -> SymPyExpression:
        # we cannot be lazy here because the goal is to create a sympy object, so arguments must be
        # recursively converted to sympy object
        match function:
            # first check if function is of SymPyConstant, where sympy_function is assumed to be a sympy function,
            # and we don't need to convert it.
            case SymPyConstant(sympy_object=sympy_function):
                return SymPyFunctionApplication.from_sympy_function_and_general_arguments(sympy_function, arguments)
            # if function is not of SymPyConstant but of Constant, then it is assumed to be a python callable
            case Constant(value=python_callable):
                # during the call, ValueError will be implicitly raised if we cannot convert
                sympy_function = _python_callable_to_sympy_function(python_callable)
                return SymPyFunctionApplication.from_sympy_function_and_general_arguments(sympy_function, arguments)
            case Variable(name=name, type=type_):
                sympy_function = sympy.Function(name)
                return SymPyFunctionApplication.from_sympy_function_and_general_arguments(sympy_function, arguments,
                                                                                          type_)
            case FunctionApplication(_, _):
                raise ValueError("The function must be a python callable.")
            case _:
                raise ValueError("Unknown case.")

    @classmethod
    def new_quantifier_expression(cls, operation: Constant, index: Variable, constraint: Expression, body: Expression,
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
                return _sympy_function_to_python_callable(value)
            except Exception as exc:
                raise ValueError(f"Cannot pythonize {value}.") from exc

    @property
    def sympy_object(self):
        return self._sympy_object

    @property
    def type_dict(self) -> Dict[sympy.Basic, ExpressionType]:
        return self._type_dict

    def internal_object_eq(self, other) -> bool:
        match other:
            case SymPyExpression(sympy_object=other_sympy_object, type_dict=other_type_dict):
                return self.sympy_object == other_sympy_object and self.type_dict == other_type_dict
            case _:
                return False

    @staticmethod
    def from_sympy_object(sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]) -> SymPyExpression:
        # Here we just try to find a type of expression for sympy object.
        if isinstance(sympy_object, sympy.Symbol):
            return SymPyVariable(sympy_object, type_dict[sympy_object])
        elif _is_sympy_value(sympy_object):
            return SymPyConstant(sympy_object, _infer_sympy_object_type(sympy_object, type_dict))
        elif _is_sympy_sum(sympy_object):
            return SymPySummation(sympy_object, type_dict)
        elif _is_sympy_integral(sympy_object):
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
        SymPyExpression.__init__(self, sympy_object, expression_type, {sympy_object: expression_type})

    @property
    def atom(self) -> str:
        return str(self._sympy_object)


class SymPyConstant(SymPyExpression, Constant):
    def __init__(self, sympy_object: sympy.Basic, expression_type: Optional[ExpressionType] = None):
        if expression_type is None:
            expression_type = _infer_sympy_object_type(sympy_object, {})
        SymPyExpression.__init__(self, sympy_object, expression_type, {})  # type_dict only records variables

    @property
    def atom(self) -> Any:
        return SymPyExpression.pythonize_value(self._sympy_object)


class SymPyFunctionApplicationInterface(SymPyExpression, FunctionApplication, ABC):
    @property
    @abstractmethod
    def native_arguments(self) -> Tuple[sympy.Basic, ...]:
        pass

    @property
    @abstractmethod
    def function_type(self) -> ExpressionType:
        pass

    @property
    def function(self) -> Expression:
        if is_sympy_uninterpreted_function(self._sympy_object.func):
            return SymPyVariable(self._sympy_object.func, self.function_type)
        else:
            return SymPyConstant(self._sympy_object.func, self.function_type)

    @property
    def arguments(self) -> List[Expression]:
        return [SymPyExpression.from_sympy_object(argument, self.type_dict)
                for argument in self.native_arguments]

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.function] + self.arguments


class SymPyFunctionApplication(SymPyFunctionApplicationInterface):
    def __new__(cls, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]):
        if sympy_object.func == sympy.Piecewise:
            return SymPyConditionalFunctionApplication(sympy_object, type_dict)
        else:
            return super().__new__(cls)

    def __init__(self, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]):
        """
        Calling by function_type=None asks this function to try to infer the function type.
        If the caller knows the function_type, it should always set function_type to a non-None value.
        This function always set type_dict[sympy_object] with the new (inferred or supplied) function_type value.
        The old value, if exists, is only used for consistency checking.
        """
        if not sympy_object.args:
            raise TypeError("not a function application.")

        if sympy_object.func in type_dict:
            # this happens iff sympy_object is an uninterpreted function, whose type cannot be inferred
            self._function_type = type_dict[sympy_object.func]
        else:
            self._function_type = _infer_sympy_function_type(sympy_object, type_dict)
        return_type = return_type_after_application(self._function_type, len(sympy_object.args))
        SymPyExpression.__init__(self, sympy_object, return_type, type_dict)

    @property
    def function_type(self) -> ExpressionType:
        return self._function_type

    @property
    def number_of_arguments(self) -> int:
        return len(self.native_arguments)

    @property
    def native_arguments(self) -> Tuple[sympy.Basic, ...]:
        """ faster than arguments """
        return self._sympy_object.args  # sympy f.args returns a tuple

    @staticmethod
    def from_sympy_function_and_general_arguments(sympy_function: sympy.Basic, arguments: List[Expression],
                                                  uninterpreted_function_type: ExpressionType = None,
                                                  ) -> SymPyExpression:
        sympy_arguments = [SymPyExpression._convert(argument) for argument in arguments]
        type_dict = _build_type_dict_from_sympy_arguments(sympy_arguments)

        if is_sympy_uninterpreted_function(sympy_function):
            # If the function is uninterpreted, we must also save its type information in type_dict,
            # since its type cannot be inferred.
            if uninterpreted_function_type is None:
                raise ValueError(f"uninterpreted function {sympy_function} has no type!")
            type_dict[sympy_function] = uninterpreted_function_type

        if sympy_function == sympy.Min or sympy_function == sympy.Max:
            # see test/sympy_test.py: test_sympy_bug()
            sympy_object = sympy_function(*[sympy_argument.sympy_object for sympy_argument in sympy_arguments],
                                          evaluate=global_parameters.sympy_evaluate)
        elif sympy_function == sympy.Piecewise:
            sympy_object = sympy_piecewise_from_if_then_else(
                *[sympy_argument.sympy_object for sympy_argument in sympy_arguments])
        else:
            with sympy.evaluate(global_parameters.sympy_evaluate):
                # If we want to preserve the symbolic structure, we need to stop evaluation by setting
                # global_parameters.sympy_evaluate to False (or Add(1,1) will be 2 in sympy).
                sympy_object = sympy_function(*[sympy_argument.sympy_object for sympy_argument in sympy_arguments])

        if global_parameters.sympy_evaluate:
            # if sympy_evaluate is True, we don't necessarily return a FunctionApplication.
            # E.g., sympy_object = (a + y) - y would be a.
            return SymPyExpression.from_sympy_object(sympy_object, type_dict)
        else:
            return SymPyFunctionApplication(sympy_object, type_dict)


def sympy_piecewise_from_if_then_else(if_: sympy.Basic, then_: sympy.Basic, else_: sympy.Basic) -> sympy.Piecewise:
    """ In Piecewise, conditional comes after clause. """
    return sympy.Piecewise((then_, if_), (else_, True))


def sympy_piecewise_to_if_then_else(piecewise: sympy.Piecewise) -> Tuple[sympy.Basic, sympy.Basic, sympy.Basic]:
    return piecewise.args[0][1], piecewise.args[0][0], piecewise.args[1][0]


def fold_sympy_piecewise(piecewise_args: List[Tuple[sympy.Basic, sympy.Basic]]) -> sympy.Piecewise:
    """ `fold` any sympy piecewise to 2-entry piecewise. E.g.,
    Piecewise((s0, c0), (s1, c1), (s2, True)) will be folded into
    Piecewise((s0, c0), (Piecewise((s1, c1), (s2, True)), True)
    """
    if len(piecewise_args) < 2:
        raise TypeError("piecewise is expected to have at least two entries")
    elif len(piecewise_args) == 2:
        return sympy.Piecewise(*piecewise_args)
    else:
        return sympy.Piecewise(piecewise_args[0], (fold_sympy_piecewise(piecewise_args[1:]), True))


class SymPyConditionalFunctionApplication(SymPyFunctionApplicationInterface):
    def __init__(self, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]):
        if sympy_object.func != sympy.Piecewise:
            raise TypeError("Can only create conditional function application when function is sympy.Piecewise.")
        if not sympy_object.args[-1][1]:  # the clause condition must be True otherwise it's not an if-then-else
            raise TypeError("Missing else clause.")
        sympy_object = fold_sympy_piecewise(sympy_object.args)
        self._then_type = _infer_sympy_object_type(sympy_object.args[0][0], type_dict)
        SymPyExpression.__init__(self, sympy_object, self._then_type, type_dict)

    @property
    def function_type(self) -> ExpressionType:
        return Callable[[bool, self._then_type, self._then_type], self._then_type]

    @property
    def function(self) -> Expression:
        return SymPyConstant(self._sympy_object.func, self.function_type)

    @property
    def number_of_arguments(self) -> int:
        return 3

    @property
    def native_arguments(self) -> Tuple[sympy.Basic, ...]:
        return sympy_piecewise_to_if_then_else(self.sympy_object)


def _context_to_variable_value_dict_helper(context: FunctionApplication,
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
        case FunctionApplication(function=Constant(value=operator.and_), arguments=arguments):
            # the conjunctive case
            for sub_context in arguments:
                variable_to_value, unknown, unsatisfiable = \
                    _context_to_variable_value_dict_helper(sub_context, variable_to_value, unknown, unsatisfiable)
        case FunctionApplication(function=Constant(value=operator.eq),
                                 arguments=[Variable(name=variable), Constant(value=value)]) | \
             FunctionApplication(function=Constant(value=operator.eq),
                                 arguments=[Constant(value=value), Variable(name=variable)]):
            # the leaf case
            if variable in variable_to_value and variable_to_value[variable] != value:
                unsatisfiable = True
            variable_to_value[variable] = value
        # all other cases makes the satisfiability unknown
        case _:
            unknown = True
    return variable_to_value, unknown, unsatisfiable


def _context_to_variable_value_dict(context: FunctionApplication) -> Tuple[Dict[str, Any], bool, bool]:
    """
    Returns a dictionary, and two booleans: first indicating whether its satisfiability is unknown, second indicating
    whether it is unsatisfiable (if its satisfiability is known)
    """
    return _context_to_variable_value_dict_helper(context, {})


class SymPyContext(SymPyFunctionApplication, Context):
    """
    SymPyContext is just a SymPyFunctionApplication, which always raises when asked for satisfiability
    since we don't know.
    We create a dictionary from the function application at initialization. """

    def __init__(self, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]):
        SymPyFunctionApplication.__init__(self, sympy_object, type_dict)
        self._dict, self._unknown, self._unsatisfiable = _context_to_variable_value_dict(self)
        if self._unsatisfiable:
            # if unsat, looking up dict for value does not make sense, as any value can be detailed
            self._dict = {}

    @property
    def unsatisfiable(self) -> bool:
        if self._unknown:
            raise Context.UnknownError()
        else:
            return self._unsatisfiable

    @property
    def dict(self) -> Dict[str, Any]:
        """ User should not write to the return value. """
        return self._dict

    @property
    def satisfiability_is_known(self) -> bool:
        return not self._unknown


class SymPySummation(SymPyExpression, QuantifierExpression):
    def __init__(self, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]):
        expression_type = _infer_sympy_object_type(sympy_object.args[0], type_dict)
        # sympy.Sum(body, (index, lower, upper))
        super().__init__(sympy_object, expression_type, type_dict)

    @property
    def is_integral(self) -> bool:
        return False

    @property
    def operation(self) -> AbelianOperation:
        return basic_add_operation(self.type)

    @property
    def index(self) -> SymPyVariable:
        variable = self.sympy_object.args[1][0]
        return SymPyVariable(variable, self.type_dict[variable])

    @property
    def lower_bound(self) -> SymPyExpression:
        lower = self.sympy_object.args[1][1]
        return SymPyExpression.from_sympy_object(lower, self.type_dict)

    @property
    def upper_bound(self) -> SymPyExpression:
        upper = self.sympy_object.args[1][2]
        return SymPyExpression.from_sympy_object(upper, self.type_dict)

    @property
    def constraint(self) -> Context:
        from .z3_expression import Z3SolverExpression
        empty_context = Z3SolverExpression()
        return empty_context & (self.index >= self.lower_bound) & (self.index <= self.upper_bound)

    @property
    def body(self) -> Expression:
        return SymPyExpression.from_sympy_object(self.sympy_object.args[0], self.type_dict)
