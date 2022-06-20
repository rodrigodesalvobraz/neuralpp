import sympy

from neuralpp.symbolic.expression import Expression, FunctionApplication, Constant, Variable
from neuralpp.symbolic.basic_expression import BasicConstant
from neuralpp.symbolic.simplifier import Simplifier
from neuralpp.symbolic.interpreter import Interpreter
from neuralpp.symbolic.sympy_expression import SymPyExpression, is_sympy_value, infer_sympy_object_type
from typing import Dict, Any


def context_to_variable_value_dict_helper(context: Expression,
                                          variable_to_value: Dict[sympy.Symbol, Any]) \
        -> Dict[sympy.Symbol, Any]:
    """
    variable_to_value: the mutable argument also serves as a return value.
    If the context has multiple assignments (e.g., x==3 and x==5), we just pick the last one.
    This does not violate our specification, since ex falso quodlibet, "from falsehood, anything follows".
    """
    match context:
        case FunctionApplication(function=Constant(value=sympy.And), arguments=arguments):
            # the conjunctive case
            for sub_context in arguments:
                variable_to_value = context_to_variable_value_dict_helper(sub_context, variable_to_value)
        case FunctionApplication(function=Constant(value=sympy.Eq),
                                 arguments=[Variable(name=lhs), Constant(value=rhs)]):
            # the leaf case
            variable_to_value[sympy.symbols(lhs)] = rhs
        # all other cases are ignored
    return variable_to_value


def context_to_variable_value_dict(context: Expression) -> \
        Dict[sympy.Symbol, int | sympy.Integer]:
    return context_to_variable_value_dict_helper(context, {})


class SymPyInterpreter(Interpreter, Simplifier):
    @staticmethod
    def _simplify_expression(expression: SymPyExpression, context: Expression) -> sympy.Basic:
        variable_value_dict = context_to_variable_value_dict(context)
        # in creation of function application, we set evaluate=False, so 1 + 2 will not evaluate
        # call simplify() evaluates that
        result = expression.sympy_object.simplify()
        for variable, value in variable_value_dict.items():
            result = result.replace(variable, value)
        return result

    def eval(self, expression: SymPyExpression, context: Expression = BasicConstant(True)):
        result = SymPyInterpreter._simplify_expression(expression, context)
        if is_sympy_value(result):
            return result
        else:
            raise RuntimeError(f"cannot evaluate to a value. The best effort result is {result}.")

    def simplify(self, expression: SymPyExpression, context: SymPyExpression = BasicConstant(True)) -> SymPyExpression:
        """
        The function calls simplify() from sympy library and wrap the result in SymPyExpression.
        """
        result = SymPyInterpreter._simplify_expression(expression, context)
        type_ = infer_sympy_object_type(result, expression.type_dict)
        # It is OK to reuse expression.type_dict since reusing names in "garbage" for a different type is not allowed.
        result = SymPyExpression.from_sympy_object(result, type_, expression.type_dict)
        # The result keeps the known type information from `expression`. E.g., though (y-y).simplify() = 0, it still
        # keeps the type of `y`. Say the old `y` is of int type, adding that `0` to another `y` of float type will
        # cause an error. That's why we need garbage_collect_type_dict() for cleaning the old, unreachable type
        # information.
        result.garbage_collect_type_dict()
        return result

