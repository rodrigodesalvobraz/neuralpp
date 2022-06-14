import sympy

from neuralpp.symbolic.expression import Expression, FunctionApplication, Constant, Variable
from neuralpp.symbolic.simplifier import Simplifier
from neuralpp.symbolic.interpreter import Interpreter
from neuralpp.symbolic.sympy_expression import SymPyExpression, is_sympy_value
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
    def eval(self, expression: SymPyExpression, context: Expression):
        variable_value_dict = context_to_variable_value_dict(context)
        # in creation of function application, we set evaluate=False, so 1 + 2 will not evaluate
        # call simplify() evaluates that
        result = expression.sympy_object.simplify()
        for variable, value in variable_value_dict.items():
            result = result.replace(variable, value)
        if is_sympy_value(result):
            return result
        else:
            raise RuntimeError(f"cannot evaluate to a value. The best effort result is {result}.")

    def simplify(self, expression: SymPyExpression, context: SymPyExpression) -> SymPyExpression:
        raise NotImplementedError("TODO")
