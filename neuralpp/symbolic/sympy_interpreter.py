import operator
import sympy

from neuralpp.symbolic.expression import Expression, FunctionApplication, Constant, Variable
from neuralpp.symbolic.basic_expression import BasicConstant
from neuralpp.symbolic.simplifier import Simplifier
from neuralpp.symbolic.interpreter import Interpreter
from neuralpp.symbolic.sympy_expression import SymPyExpression, is_sympy_value, infer_sympy_object_type
from neuralpp.symbolic.expression import ExpressionType
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
        case FunctionApplication(function=Constant(value=operator.and_), arguments=arguments):
            # the conjunctive case
            for sub_context in arguments:
                variable_to_value = context_to_variable_value_dict_helper(sub_context, variable_to_value)
        case FunctionApplication(function=Constant(value=operator.eq),
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
    def _simplify_expression(expression: sympy.Basic, context: Expression) -> sympy.Basic:
        variable_value_dict = context_to_variable_value_dict(context)
        # in creation of function application, we set evaluate=False, so 1 + 2 will not evaluate
        # call simplify() evaluates that
        result = expression
        if not variable_value_dict:
            result = result.simplify()
        else:
            for variable, value in variable_value_dict.items():
                result = result.replace(variable, value)
        return result

    def eval(self, expression: SymPyExpression, context: Expression = BasicConstant(True)):
        result = SymPyInterpreter._simplify_expression(expression.sympy_object, context)
        if is_sympy_value(result):
            return result
        else:
            raise RuntimeError(f"cannot evaluate to a value. The best effort result is {result}.")

    @staticmethod
    def purge_type_dict(type_dict: Dict[sympy.Basic, ExpressionType], sympy_object: sympy.Basic) -> \
            Dict[sympy.Basic, ExpressionType]:
        """
        Assumes all variables (including uninterpreted functions) used in sympy_object is in type_dict.
        Returns a new type dict that only contains the keys that's used in sympy_object.
        """
        result = {}
        for key, value in type_dict.items():
            if sympy_object.has(key):
                result[key] = value
        return result

    def simplify(self, expression: Expression, context: Expression = BasicConstant(True)) -> SymPyExpression:
        """
        The function calls simplify() from sympy library and wrap the result in SymPyExpression.
        """
        if not isinstance(expression, SymPyExpression):
            expression = SymPyExpression.convert(expression)

        simplified_sympy_expression = SymPyInterpreter._simplify_expression(expression.sympy_object, context)
        # The result keeps the known type information from `expression`. E.g., though (y-y).simplify() = 0, it still
        # keeps the type of `y`. Delete these redundant types.
        type_dict = SymPyInterpreter.purge_type_dict(expression.type_dict, simplified_sympy_expression)
        result_expression = SymPyExpression.from_sympy_object(simplified_sympy_expression, type_dict)
        return result_expression
