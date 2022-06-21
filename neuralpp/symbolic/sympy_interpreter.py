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
    def _updated_type_dict_by_context(type_dict: Dict[sympy.Basic, ExpressionType], context: Expression) -> \
            Dict[sympy.Basic, ExpressionType]:
        """ argument type_dict is not modified, instead, a new dict is created as a return value"""
        result = {}
        for k, v in type_dict.items():
            new_key = SymPyInterpreter._simplify_expression(k, context)
            if new_key == k or new_key not in type_dict:
                # if skip the update if not(new_key == k or new_key not in type_dict), i.e,
                # if (i) new_key is a new key (has been simplified()) and (ii) new_key exists in type_dict.
                # From (i) and (ii), k must be a function application (if k is a variable and simpified, new_key would
                # be constant and will not be in type_dict). Based on the type of new_key, there's two cases:
                # #1 if new_key is a variable: then new_key should be discarded as v is of function type.
                # #2 if new_key is a function application: old and new types should be same. just discard the new_key.
                result[new_key] = v
        return result

    def simplify(self, expression: Expression, context: Expression = BasicConstant(True)) -> SymPyExpression:
        """
        The function calls simplify() from sympy library and wrap the result in SymPyExpression.
        """
        if not isinstance(expression, SymPyExpression):
            expression = SymPyExpression.convert(expression)

        result = SymPyInterpreter._simplify_expression(expression.sympy_object, context)
        # if "x+y" is simplified to "x+2", how do we find out the type of that "+"? We need to update the type_dict
        # by also doing the replacement. If we have "x+(y+y)", the type_dict would be
        #   {x:int, y:int, y+y:int->int->int, x+(y+y):int->int->int}
        # by a context of {y:2}, we'd have
        #   {x:int, 2:int, 4:int->int->int, x+4:int->int->int} (entry 2 and 4 will be GCed)
        type_dict = SymPyInterpreter._updated_type_dict_by_context(expression.type_dict, context)
        type_ = type_dict[result]
        result_expression = SymPyExpression.from_sympy_object(result, type_, type_dict)
        # The result keeps the known type information from `expression`. E.g., though (y-y).simplify() = 0, it still
        # keeps the type of `y`. Say the old `y` is of int type, adding that `0` to another `y` of float type will
        # cause an error. That's why we need garbage_collect_type_dict() for cleaning the old, unreachable type
        # information.
        result_expression.garbage_collect_type_dict()
        return result_expression
