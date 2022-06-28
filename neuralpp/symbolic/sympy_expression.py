from __future__ import annotations

import typing
from typing import List, Any, Type, Callable, Tuple, Optional, Dict
from abc import ABC

import sympy
from sympy import abc
import operator
import builtins
import fractions

from functools import cached_property
from neuralpp.symbolic.expression import Expression, FunctionApplication, Variable, Constant, \
    FunctionNotTypedError, NotTypedError, return_type_after_application, ExpressionType, Context
from neuralpp.symbolic.basic_expression import infer_python_callable_type
from neuralpp.util.util import update_consistent_dict


# In this file's doc, I try to avoid the term `sympy expression` because it could mean both sympy.Expr (or sympy.Basic)
# and SymPyExpression. I usually use "sympy object" to refer to the former and "expression" to refer to the latter.


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
        case _:
            # We can look up type_dict for variable like `symbol("x")`.
            # A variable can also be an uninterpreted function `sympy.core.function.UndefinedFunction('f')`
            try:
                return type_dict[sympy_object]
            except KeyError:  # if it's not in type_dict, try figure out ourselves
                """ 
                Here, sympy_object must be function applications like 'x+y'
                """
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


sympy_Sub = sympy.Lambda((abc.x, abc.y), abc.x-abc.y)
sympy_Neg = sympy.Lambda((abc.x,), -abc.x)  # "lambda x: (-1)*x"
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
    # arithmetic
    (operator.add, sympy.Add),
    (operator.mul, sympy.Mul),
    (operator.pow, sympy.Pow),
    (operator.sub, sympy_Sub),
    (operator.neg, sympy_Neg),
    # min/max
    (builtins.min, sympy.Min),
    (builtins.max, sympy.Max),
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
        raise ValueError(f"SymPy function {sympy_function} is not recognized.")


def _python_callable_to_sympy_function(python_callable: Callable) -> sympy.Basic:
    try:
        return python_callable_sympy_function_dict[python_callable]
    except KeyError:
        raise ValueError(f"Python callable {python_callable} is not recognized.")


def _is_sympy_value(sympy_object: sympy.Basic) -> bool:
    return isinstance(sympy_object, sympy.Number) or \
           isinstance(sympy_object, sympy.logic.boolalg.BooleanAtom)


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
            except Exception:
                raise ValueError(f"SymPyConstant does not support {type(value)}: "
                                 f"unable to turn into a sympy representation internally")
        return SymPyConstant(sympy_object, type_)

    @classmethod
    def new_variable(cls, name: str, type_: ExpressionType) -> SymPyVariable:
        # if a string contains a whitespace it'll be treated as multiple variables in sympy.symbols
        if ' ' in name:
            raise ValueError(f"`{name}` should not contain a whitespace!")
        sympy_var = sympy.symbols(name)
        return SymPyVariable(sympy_var, type_)

    @classmethod
    def new_function_application(cls, function: Expression, arguments: List[Expression]) -> SymPyFunctionApplication:
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
            case Variable(name=name):
                raise ValueError(f"Cannot create a SymPyExpression from uninterpreted function {name}")
            case FunctionApplication(_, _):
                raise ValueError("The function must be a python callable.")
            case _:
                raise ValueError("Unknown case.")

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
            except Exception:
                raise ValueError(f"Cannot pythonize {value}.")

    @property
    def sympy_object(self):
        return self._sympy_object

    @property
    def type_dict(self) -> Dict[sympy.Basic, ExpressionType]:
        return self._type_dict

    def __eq__(self, other) -> bool:
        match other:
            case SymPyExpression(sympy_object=other_sympy_object, type_dict=other_type_dict):
                return self.sympy_object == other_sympy_object and self.type_dict == other_type_dict
            case _:
                return False

    @staticmethod
    def from_sympy_object(sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, Expression]) -> SymPyExpression:
        # Here we just try to find a type of expression for sympy object.
        if isinstance(sympy_object, sympy.Symbol):
            return SymPyVariable(sympy_object, type_dict[sympy_object])
        elif _is_sympy_value(sympy_object):
            return SymPyConstant(sympy_object, _infer_sympy_object_type(sympy_object, type_dict))
        else:
            return SymPyFunctionApplication(sympy_object, type_dict)

    @classmethod
    def convert(cls, from_expression: Expression) -> SymPyExpression:
        return cls._convert(from_expression)


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


class SymPyFunctionApplication(SymPyExpression, FunctionApplication):
    def __init__(self, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]):
        """
        Calling by function_type=None asks this function to try to infer the function type.
        If the caller knows the function_type, it should always set function_type to a non-None value.
        This function always set type_dict[sympy_object] with the new (inferred or supplied) function_type value.
        The old value, if exists, is only used for consistency checking.
        """
        if not sympy_object.args:
            raise TypeError("not a function application.")

        self._function_type = _infer_sympy_function_type(sympy_object, type_dict)
        return_type = return_type_after_application(self._function_type, len(sympy_object.args))
        SymPyExpression.__init__(self, sympy_object, return_type, type_dict)

    @property
    def number_of_arguments(self) -> int:
        return len(self.native_arguments)

    @property
    def function(self) -> Expression:
        return SymPyConstant(self._sympy_object.func, self._function_type)

    @property
    def arguments(self) -> List[Expression]:
        return [SymPyExpression.from_sympy_object(argument, self.type_dict)
                for argument in self.native_arguments]

    @property
    def native_arguments(self) -> Tuple[sympy.Basic, ...]:
        """ faster than arguments """
        return self._sympy_object.args  # sympy f.args returns a tuple

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.function] + self.arguments

    @staticmethod
    def from_sympy_function_and_general_arguments(sympy_function: sympy.Basic, arguments: List[Expression]) -> \
            SymPyFunctionApplication:
        sympy_arguments = [SymPyExpression._convert(argument) for argument in arguments]
        type_dict = _build_type_dict_from_sympy_arguments(sympy_arguments)

        # Stop evaluation, otherwise Add(1,1) will be 2 in sympy.
        if sympy_function == sympy.Min or sympy_function == sympy.Max:
            # see test/sympy_test.py: test_sympy_bug()
            sympy_object = sympy_function(*[sympy_argument.sympy_object for sympy_argument in sympy_arguments],
                                          evaluate=False)
            return SymPyFunctionApplication(sympy_object, type_dict)
        else:
            with sympy.evaluate(False):
                sympy_object = sympy_function(*[sympy_argument.sympy_object for sympy_argument in sympy_arguments])
                return SymPyFunctionApplication(sympy_object, type_dict)


def _context_to_variable_value_dict_helper(context: FunctionApplication,
                                           variable_to_value: Dict[str, Any]) -> Dict[str, Any]:
    """
    variable_to_value: the mutable argument also serves as a return value.
    If the context has multiple assignments (e.g., x==3 and x==5), we just pick the last one.
    This does not violate our specification, since ex falso quodlibet, "from falsehood, anything follows".
    """
    match context:
        case FunctionApplication(function=Constant(value=operator.and_), arguments=arguments):
            # the conjunctive case
            for sub_context in arguments:
                variable_to_value = _context_to_variable_value_dict_helper(sub_context, variable_to_value)
        case FunctionApplication(function=Constant(value=operator.eq),
                                 arguments=[Variable(name=lhs), Constant(value=rhs)]):
            # the leaf case
            variable_to_value[lhs] = rhs
        # all other cases are ignored
    return variable_to_value


def _context_to_variable_value_dict(context: FunctionApplication) -> Dict[str, Any]:
    return _context_to_variable_value_dict_helper(context, {})


class SymPyContext(SymPyFunctionApplication, Context):
    """
    SymPyContext is just a SymPyFunctionApplication, which always raises when asked for satisfiability
    since we don't know.
    We create a dictionary from the function application at the first time `dict()` property is called. """
    @property
    def unsatisfiable(self) -> bool:
        raise Context.UnknownError()

    @cached_property
    def dict(self) -> Dict[str, Any]:
        return _context_to_variable_value_dict(self)

