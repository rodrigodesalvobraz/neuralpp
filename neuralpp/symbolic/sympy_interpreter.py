import operator
import sympy

from neuralpp.symbolic.expression import Expression, FunctionApplication, Constant, Variable, Context
from neuralpp.symbolic.basic_expression import TrueContext
from neuralpp.symbolic.simplifier import Simplifier
from neuralpp.symbolic.interpreter import Interpreter
from neuralpp.symbolic.sympy_expression import SymPyExpression, _is_sympy_value, _infer_sympy_object_type
from neuralpp.symbolic.expression import ExpressionType
from typing import Dict, Any


class SymPyInterpreter(Interpreter, Simplifier):
    @staticmethod
    def _simplify_expression(expression: sympy.Basic, context: Context) -> sympy.Basic:
        result = expression
        if not context.dict:
            # in creation of function application, we set evaluate=False, so 1 + 2 will not evaluate
            # call simplify() evaluates that
            result = result.simplify()
        else:
            for variable, value in context.dict.items():
                result = result.replace(sympy.symbols(variable), value)
        return result

    def eval(self, expression: SymPyExpression, context: Context = TrueContext()):
        result = SymPyInterpreter._simplify_expression(expression.sympy_object, context)
        if _is_sympy_value(result):
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

    def simplify(self, expression: Expression, context: Context = TrueContext()) -> SymPyExpression:
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
