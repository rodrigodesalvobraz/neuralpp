import sympy

from .normalizer import Normalizer
from .sympy_interpreter import SymPyInterpreter
from .context_simplifier import ContextSimplifier
from .expression import Expression, Constant, Variable, FunctionApplication, QuantifierExpression, Context, \
    AbelianOperation
from .z3_expression import Z3SolverExpression
from .constants import basic_true, basic_false, if_then_else
from .parameters import sympy_evaluate
from .basic_expression import BasicQuantifierExpression, BasicExpression, BasicConstant
from .eliminator import Eliminator
import neuralpp.symbolic.functions as functions
from typing import List
from .sympy_expression import SymPyExpression, make_piecewise
from .evaluator import Evaluator

_simple_simplifier = SymPyInterpreter()
_simplifier = ContextSimplifier()


def conditional_given_context(condition: Expression, then: Expression, else_: Expression, context: Z3SolverExpression):
    """
    """
    if context.is_known_to_imply(condition):
        return then
    elif context.is_known_to_imply(~condition):
        return else_
    else:
        return if_then_else(condition, then, else_,)


def _split_into_literals(condition: Expression) -> List[Expression]:
    from .sympy_interpreter import SymPyInterpreter
    import operator
    condition = SymPyInterpreter().simplify(condition)  # get an DNF
    match condition:
        case FunctionApplication(function=Constant(value=operator.or_)):
            raise NotImplementedError("Not expecting OR")
        case FunctionApplication(function=Constant(value=operator.and_), arguments=arguments):
            return arguments
        case _:
            return [condition]


def _check_no_conditional(argument: Expression):
    match argument:
        case FunctionApplication(function=Constant(value=sympy.Piecewise)):
            raise AttributeError("WRONG")
        case _:
            for subexpression in argument.subexpressions:
                _check_no_conditional(subexpression)


class GeneralNormalizer(Normalizer):
    def __init__(self, evaluator: Evaluator = None):
        if evaluator is None:
            self.evaluator = Evaluator()
        else:
            self.evaluator = evaluator
        self._eliminator = Eliminator(self.evaluator)

    def normalize(self, expression: Expression, context: Z3SolverExpression) -> Expression:
        with sympy_evaluate(True):
            return self._normalize(expression, context)

    def _normalize(self, expression: Expression, context: Z3SolverExpression, body_is_normalized: bool = False) -> Expression:
        """
        The function assumes context is satisfiable, otherwise the behavior is undefined.
        The input expression can be arbitrarily structured. The result expression is guaranteed to be
        `quantifiers-at-leaves`: quantifiers are at leaves and do not contain conditional function in their bodies.
        `body_is_normalized`: used by QuantifierExpression only to indicate body is normalized
        """
        assert not context.unsatisfiable
        match expression:
            case Constant():
                return expression
            case FunctionApplication(function=Constant(value=sympy.Piecewise),
                                     arguments=arguments):
                new_conditions = []
                new_expressions = []
                for expression, condition in arguments:
                    if context.is_known_to_imply(condition):
                        print(f"shortcut: {SymPyExpression.convert(context).sympy_object} -> {SymPyExpression.convert(condition).sympy_object}")
                        return self._normalize(expression, context)
                    elif context.is_known_to_imply(~condition):
                        print(f"eliminate: {SymPyExpression.convert(condition).sympy_object}")
                        pass
                    else:
                        new_conditions.append(condition)
                        new_expressions.append(self._normalize(expression, context & condition))
                # print(f"len: {len(new_expressions)}, {len(new_conditions)}")
                return make_piecewise(new_conditions, new_expressions)

            case FunctionApplication(function=Constant(value=functions.conditional),
                                     arguments=[condition, then, else_]):
                raise
            case Expression(type=type_) if type_ == bool:
                return expression
            case Variable():
                return expression
            case FunctionApplication(function=function, arguments=arguments):
                return self._normalize_function_application(function, arguments, context)
            case QuantifierExpression(operation=operation, index=index, constraint=constraint, body=body,
                                      is_integral=is_integral):
                if context.contains(index):
                    raise ValueError(f"context {context} should not contain index {index}")
                if context.is_known_to_imply(~constraint):
                    return operation.identity

                if body_is_normalized:
                    normalized_body = body
                else:
                    normalized_body = self._normalize(body, context & constraint)
                    # print(f"normalized_body {SymPyExpression.convert(normalized_body).sympy_object}")
                    # print(f"normalized_body {SymPyExpression.convert(normalized_body).sympy_object.args}")
                    # print(f"normalized_body {len(SymPyExpression.convert(normalized_body).sympy_object.args)}")
                    # print(f"normalized_body {SymPyExpression.convert(normalized_body).sympy_object.args[0]}")
                    # print(f"normalized_body {SymPyExpression.convert(normalized_body).sympy_object.args[1]}")

                match normalized_body:
                    case FunctionApplication(function=Constant(value=sympy.Piecewise),
                                             arguments=arguments):
                        if arguments[0][1].contains(index):
                            elements = []
                            for expression, condition in arguments:
                                assert condition.contains(index)
                                elements.append(self._normalize(
                                    BasicQuantifierExpression(operation, index, constraint & condition, expression,
                                                              is_integral), context, body_is_normalized=True))
                            with self.evaluator.log_section("symbolic addition"):
                                result = SymPyExpression.new_function_application(operation, elements)
                            # print(f'result={result.sympy_object}')
                            # return _simple_simplifier.simplify(result)
                            return result
                        else:
                            new_expressions = []
                            conditions = []
                            for expression, condition in arguments:
                                assert not condition.contains(index)
                                conditions.append(condition)
                                new_expressions.append(self._normalize(BasicQuantifierExpression(operation, index, constraint, expression, is_integral), context & condition, body_is_normalized=True))
                            with self.evaluator.log_section("make piecewise"):
                                return make_piecewise(conditions, new_expressions)

                    case FunctionApplication(function=Constant(value=functions.conditional),
                                             arguments=[condition, then, else_]):
                        raise
                    case _:
                        return self._eliminate(operation, index, constraint, normalized_body, is_integral, context)

    def _normalize_function_application(self, function: Expression,
                                        arguments: List[Expression],
                                        context: Z3SolverExpression, i: int = 0) -> Expression:
        if i >= len(arguments):
            return function(*arguments)
            # return _simplifier.simplify(function(*arguments), context)
        arguments[i] = self._normalize(arguments[i], context)
        return self._move_down_and_normalize(function, arguments, context, i)

    def _move_down_and_normalize(self,
                                 function: Expression,
                                 arguments: List[Expression],
                                 context: Z3SolverExpression, i: int) -> Expression:
        """
        assume i < len(arguments);
        assume all {arguments[j] | j < i} does not contain if-then-else (and thus normalized);
        assume arguments[i] has been normalized.
        move down the f to the leaves. Then replace every leaf f with its normalized version
        """
        match arguments[i]:
            case FunctionApplication(function=Constant(value=sympy.Piecewise),
                                     arguments=piecewise_arguments):
                new_conditions = []
                new_expressions = []
                for expression, condition in piecewise_arguments:
                    if context.is_known_to_imply(condition):
                        print(
                            f"shortcut: {SymPyExpression.convert(context).sympy_object} -> {SymPyExpression.convert(condition).sympy_object}")
                        arguments[i] = expression
                        return self._move_down_and_normalize(function, arguments, context & condition, i)
                    elif context.is_known_to_imply(~condition):
                        print(f"eliminate: {SymPyExpression.convert(condition).sympy_object}")
                        pass
                    else:
                        new_conditions.append(condition)
                        new_arguments = arguments[:]
                        new_arguments[i] = expression
                        new_expressions.append(
                            self._move_down_and_normalize(function, new_arguments, context & condition, i))
                # print(f"moving down {function} {i} {len(piecewise_arguments)}")
                return make_piecewise(new_conditions, new_expressions)
            case FunctionApplication(function=Constant(value=functions.conditional)):
                raise
            case _:
                _check_no_conditional(arguments[i])
                return self._normalize_function_application(function, arguments, context, i + 1)

    def _eliminate(self, operation: AbelianOperation, index: Variable, constraint: Context, body: Expression,
                   is_integral: bool, context: Z3SolverExpression) -> Expression:
        """
        Eliminates all quantifiers by doing the "summation" (or to use a Computer Science term, "reduction").
        In particular, we expect body and result to be normalized quantifiers-at-leaves.

        Future work: more complicated elimination algorithm, which actually tries to `eliminate` quantifiers.
        In general 2 directions for improvement:
        1. supports more operations (add, multiply, and, or, ...)
        2. supports multiple intervals & complicated constraints (e.g, 1 <= x <= 100, x != y)
        """
        if isinstance(body, FunctionApplication) and body.function.value == functions.conditional:
            raise AttributeError("WHAT")
        if context.is_known_to_imply(~constraint):
            return operation.identity
        # print(f"eliminating: {body}")
        result = self._eliminator.eliminate(operation, index, constraint, body, is_integral, context)
        # print(f"done")
        return result
        # return _simplifier.simplify(result, context)
