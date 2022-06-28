from typing import List, Any, Optional, Type, Callable, Dict
import operator

from neuralpp.inference.graphical_model.representation.factor.atomic_factor import AtomicFactor
from neuralpp.inference.graphical_model.variable.discrete_variable import DiscreteVariable
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.symbolic.basic_expression import BasicConstant, BasicVariable, BasicFunctionApplication
from neuralpp.symbolic.expression import Expression, FunctionApplication
from neuralpp.symbolic.sympy_interpreter import SymPyInterpreter
from neuralpp.util.util import union, join

class SymbolicFactor(AtomicFactor):
    def __init__(self, variables: List[Variable], expression: Expression):
        super().__init__(variables)
        self.expression = expression

    def new_instance(self, variables: List[Variable], expression: Expression):
        return type(self)(variables, expression)

    def assignments(self):
        return DiscreteVariable.assignments_product(self.variables)

    def _variable_type(self, variable: Variable) -> type:
        if (isinstance(variable, IntegerVariable)):
            return int
        else:
            return float

    def _dict_to_context(self, assignment_dict: dict) -> Expression:
        result = BasicConstant(True)
        type_dict = {}
        for k, v in assignment_dict.items():
            symbol = BasicVariable(k.name, self._variable_type(v))
            eq_operator = BasicConstant(operator.eq, Callable[[type(v), type(v)], bool])
            eq_expression = BasicFunctionApplication(eq_operator, [symbol, BasicConstant(v, type(v))])

            and_operator = BasicConstant(operator.and_, Callable[[bool, bool], bool])
            result = BasicFunctionApplication(and_operator, [result, eq_expression])
        return result

    def dict_to_context_result(self,  assignment_dict: dict) -> Expression:
        return self._dict_to_context(assignment_dict)

    def condition_on_non_empty_dict(self, assignment_dict: Dict[Variable, Any]):
        # TODO: handle batch cases
        non_conditioned_variables = [
            v
            for v in self.variables
            if v not in assignment_dict or isinstance(assignment_dict[v], slice)
        ]

        interpreter = SymPyInterpreter()
        context = self._dict_to_context(assignment_dict)
        conditioned_expression = interpreter.simplify(self.expression, context)

        return self.new_instance(non_conditioned_variables, conditioned_expression)


    def call_after_validation(self, assignment_dict, assignment_values):
        interpreter = SymPyInterpreter()
        context = self._dict_to_context(assignment_dict)

        return interpreter.eval(self.expression, context)

    def mul_by_non_identity(self, other):
        """Multiplies factors so that (f1 * f2)(assignment) = f1(assignment)*f2(assignment)
           Multiply the two expressions together to create a new expression and create a new symbolic factor with that"""
        if not isinstance(other, SymbolicFactor):
            raise Exception(
                f"Multiplication of SymbolicFactor to factors other than SymbolicFactor is not implemented. "
                f"Got {type(other)}"
            )
        combined_variables = union([self.variables, other.variables])
        combined_expression = self.expression * other.expression

        return self.new_instance(combined_variables, combined_expression)

    def sum_out_variable(self, variable: Variable):
        result_variables = [v for v in self.variables if v != variable]

        interpreter = SymPyInterpreter()
        result_expression = None
        for a in variable.assignments():
            symbol = BasicVariable(variable.name, type(a))
            eq_operator = BasicConstant(operator.eq, Callable[[type(a), type(a)], bool])
            context = BasicFunctionApplication(eq_operator, [symbol, BasicConstant(a, type(a))])
            simplified_expression = interpreter.simplify(self.expression, context)

            if result_expression == None:
                result_expression = simplified_expression
            else:
                result_expression = result_expression + simplified_expression

        result_expression = interpreter.simplify(result_expression)
        return self.new_instance(result_variables, result_expression)

    def argmax(self):
        raise NotImplementedError("TODO")

    def normalize(self):
        interpreter = SymPyInterpreter()
        sum_expression = self.sum_out_variables(self.variables).expression

        result_expression = self.expression / sum_expression
        result_expression = interpreter.simplify(result_expression)
        return self.new_instance(self.variables, result_expression)

    def randomize(self):
        raise NotImplementedError("TODO")

    def randomized_copy(self):
         raise NotImplementedError("TODO")

    def __eq__(self, other):
        """
        Compares factors by checking if they have the same variables
        (after appropriate permutation if variables are not in the same order).
        """
        if isinstance(other, SymbolicFactor):
            return (self.variables == other.variables and
                self.expression == other.expression)
        else:
            raise Exception(
                f"Comparison of SymbolicFactor to factors other than SymbolicFactor is not implemented. "
                f"Got {type(other)}"
            )

    def __repr__(self):
        return "Factor on (" + join(self.variables) + "): " + repr(self.expression)

    def __str__(self):
        return "Factor on (" + join(self.variables) + "): " + str(self.expression)
