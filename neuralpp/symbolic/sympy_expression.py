from __future__ import annotations
from typing import List, Any, Type, Callable, Tuple, Optional, Dict
from abc import ABC

import sympy
import operator
import builtins
import fractions
from neuralpp.symbolic.expression import Expression, FunctionApplication, Variable, Constant, \
    FunctionNotTypedError, NotTypedError, return_type_after_application, ExpressionType
from neuralpp.symbolic.basic_expression import infer_python_callable_type
from neuralpp.util.util import update_consistent_dict


# In this file's doc, I try to avoid the term `sympy expression` because it could mean both sympy.Expr (or sympy.Basic)
# and SymPyExpression. I usually use "sympy object" to refer to the former and "expression" to refer to the latter.


def infer_sympy_object_type(sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType]) -> ExpressionType:
    """
    type_dict can be, for example, {a: int, b: int, c: float, a+b:int->int->int, (a+b)+c:int->float->float}.
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
            # It is obvious that we can look up type_dict for symbols like `symbol("x")`.
            # We can also look up for function application, such as `a+b`, where type_dict records the
            # function type (instead of return type) of the function used in the function application.
            try:
                return type_dict[sympy_object]
            except KeyError:  # if it's not in type_dict, try figure out ourselves
                return infer_python_callable_type(sympy_function_to_python_callable(sympy_object))


python_callable_and_sympy_function_relation = [
    # boolean operation
    (operator.and_, sympy.And),
    (operator.or_, sympy.Or),
    (operator.not_, sympy.Not),
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


def sympy_function_to_python_callable(sympy_function: sympy.Basic) -> Callable:
    try:
        return sympy_function_python_callable_dict[sympy_function]
    except KeyError:
        raise ValueError(f"SymPy function {sympy_function} is not recognized.")


def python_callable_to_sympy_function(python_callable: Callable) -> sympy.Basic:
    try:
        return python_callable_sympy_function_dict[python_callable]
    except KeyError:
        raise ValueError(f"Python callable {python_callable} is not recognized.")


def is_sympy_value(sympy_object: sympy.Basic) -> bool:
    return isinstance(sympy_object, sympy.Number) or \
           isinstance(sympy_object, sympy.logic.boolalg.BooleanAtom)


def sympy_object_to_expression(sympy_object: sympy.Basic, argument_type: Expression,
                               type_dict: Dict[sympy.Basic, Expression]) -> SymPyExpression:
    # Here we just try to find a type of expression for sympy object.
    if type(sympy_object) == sympy.Symbol:
        return SymPyVariable(sympy_object, argument_type)
    elif is_sympy_value(sympy_object):
        return SymPyConstant(sympy_object, argument_type)
    else:
        return SymPyFunctionApplication(sympy_object, type_dict, type_dict[sympy_object])


def build_type_dict(sympy_arguments: SymPyExpression, type_dict: Dict[sympy.Basic, Expression]) -> None:
    update_consistent_dict(type_dict, sympy_arguments.type_dict)


def build_type_dict_from_sympy_arguments(sympy_arguments: List[SymPyExpression]) -> Dict[sympy.Basic, Expression]:
    """
    Assumption: each element in sympy_arguments has a proper type_dict.
    Returns: a proper type_dict with these arguments joint
    """
    result = {}
    for sympy_argument in sympy_arguments:
        build_type_dict(sympy_argument, result)
    return result


class SymPyExpression(Expression, ABC):
    def __init__(self, sympy_object: sympy.Basic, expression_type: Expression,
                 type_dict: Dict[sympy.Basic, Expression]):
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
        elif type(value) == bool:
            sympy_object = sympy.S.true if value else sympy.S.false
        elif type(value) == int:
            sympy_object = sympy.Integer(value)
        elif type(value) == float:
            sympy_object = sympy.Float(value)
        elif type(value) == fractions.Fraction:
            sympy_object = sympy.Rational(value)
        elif type(value) == str:
            sympy_object = sympy.core.function.UndefinedFunction(value)
            if type_ is None:
                raise FunctionNotTypedError
        else:
            try:
                sympy_object = python_callable_to_sympy_function(value)
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
            case SymPyConstant(sympy_object=sympy_function, type=function_type):
                return SymPyFunctionApplication.from_sympy_function_and_general_arguments(
                    sympy_function, function_type, arguments)
            # if function is not of SymPyConstant but of Constant, then it is assumed to be a python callable
            case Constant(value=python_callable, type=function_type):
                # during the call, ValueError will be implicitly raised if we cannot convert
                sympy_function = python_callable_to_sympy_function(python_callable)
                return SymPyFunctionApplication.from_sympy_function_and_general_arguments(
                    sympy_function, function_type, arguments)
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
                return sympy_function_to_python_callable(value)
            except Exception:
                raise ValueError(f"Cannot pythonize {value}.")

    @property
    def sympy_object(self):
        return self._sympy_object

    @property
    def type_dict(self) -> Dict[sympy.Basic, Expression]:
        return self._type_dict

    def __eq__(self, other) -> bool:
        match other:
            case SymPyExpression(sympy_object=other_sympy_object, type_dict=other_type_dict):
                return self.sympy_object == other_sympy_object and self.type_dict == other_type_dict
            case _:
                return False


class SymPyVariable(SymPyExpression, Variable):
    def __init__(self, sympy_object: sympy.Basic, expression_type: Expression):
        SymPyExpression.__init__(self, sympy_object, expression_type, {sympy_object: expression_type})

    @property
    def atom(self) -> str:
        return str(self._sympy_object)


class SymPyConstant(SymPyExpression, Constant):
    def __init__(self, sympy_object: sympy.Basic, expression_type: Optional[Expression] = None):
        if expression_type is None:
            expression_type = infer_sympy_object_type(sympy_object, {})
        SymPyExpression.__init__(self, sympy_object, expression_type, {sympy_object: expression_type})

    @property
    def atom(self) -> Any:
        return SymPyExpression.pythonize_value(self._sympy_object)


class SymPyFunctionApplication(SymPyExpression, FunctionApplication):
    def __init__(self, sympy_object: sympy.Basic, type_dict: Dict[sympy.Basic, ExpressionType],
                 function_type: Optional[Expression] = None):
        if function_type is None:
            # it's useless to supply type_dict since we are looking for self.type, which is not set in type_dict yet.
            function_type = infer_sympy_object_type(sympy_object.func, {})

        if len(sympy_object.args) == 0:
            raise TypeError("not a function application.")

        if sympy_object not in type_dict:
            type_dict[sympy_object] = function_type
        else:
            if type_dict[sympy_object] != function_type:
                raise ValueError(f"Inconsistent type for {sympy_object}: {function_type} and "
                                 f"{type_dict[sympy_object]}.")

        self._function_type = function_type
        return_type = return_type_after_application(function_type, len(sympy_object.args))
        SymPyExpression.__init__(self, sympy_object, return_type, type_dict)

    @property
    def number_of_arguments(self) -> int:
        return len(self.native_arguments)

    @property
    def function(self) -> Expression:
        return SymPyConstant(self._sympy_object.func, self._function_type)

    @property
    def arguments(self) -> List[Expression]:
        return [sympy_object_to_expression(argument, infer_sympy_object_type(argument, self.type_dict), self.type_dict)
                for argument in self.native_arguments]

    @property
    def native_arguments(self) -> Tuple[sympy.Basic, ...]:
        """ faster than arguments """
        return self._sympy_object.args  # sympy f.args returns a tuple

    @property
    def subexpressions(self) -> List[Expression]:
        return [self.function] + self.arguments

    @staticmethod
    def from_sympy_function_and_general_arguments(sympy_function: sympy.Basic, function_type: Expression,
                                                  arguments: List[Expression]) -> SymPyFunctionApplication:
        sympy_arguments = [SymPyExpression._convert(argument) for argument in arguments]
        type_dict = build_type_dict_from_sympy_arguments(sympy_arguments)
        sympy_object = sympy_function(*[sympy_argument.sympy_object for sympy_argument in sympy_arguments],
                                      evaluate=False)  # Stop evaluation, otherwise Add(1,1) will be 2 in sympy
        return SymPyFunctionApplication(sympy_object, type_dict, function_type)
