from typing import List

import sympy

import neuralpp.symbolic.functions as functions
from neuralpp.util.util import distinct_pairwise
from .basic_expression import BasicQuantifierExpression, BasicExpression
from .constants import basic_true, basic_false, if_then_else
from .context_simplifier import ContextSimplifier
from .eliminator import Eliminator
from .expression import Expression, Constant, Variable, FunctionApplication, QuantifierExpression, Context, \
    AbelianOperation
from .normalizer import Normalizer
from .parameters import sympy_evaluate
from .profiler import Profiler
from .sympy_expression import SymPyExpression, make_piecewise
from .z3_expression import Z3SolverExpression

_simplifier = ContextSimplifier()


def conditional_given_context(condition: Expression, then: Expression, else_: Expression, context: Z3SolverExpression):
    """
    """
    if context.is_known_to_imply(condition):
        return then
    elif context.is_known_to_imply(~condition):
        return else_
    else:
        return if_then_else(condition, then, else_, )


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
        case FunctionApplication(function=Constant(value=sympy.Piecewise)) | FunctionApplication(
             function=Constant(value=functions.conditional)):
            raise AttributeError(f"Expression contains conditional/piecewise expression {argument}!")
        case _:
            for subexpression in argument.subexpressions:
                _check_no_conditional(subexpression)


class GeneralNormalizer(Normalizer):
    def __init__(self, profiler: Profiler = None):
        if profiler is None:
            self.profiler = Profiler()
        else:
            self.profiler = profiler
        self._eliminator = Eliminator(self.profiler)

    def normalize(self, expression: Expression, context: Z3SolverExpression, simplify: bool = False) -> Expression:
        with sympy_evaluate(True):
            result = self._normalize(expression, context)
            if simplify:
                return _simplifier.simplify(result, context)
            else:
                return result

    def _normalize(self, expression: Expression, context: Z3SolverExpression,
                   body_is_normalized: bool = False) -> Expression:
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
                with self.profiler.profile_section("piecewise-normalization"):
                    new_arguments = []
                    for expression, condition in distinct_pairwise(arguments):
                        if context.is_known_to_imply(condition):
                            print(
                                f"normalize shortcut: {SymPyExpression.convert(context).sympy_object} -> "
                                f"{SymPyExpression.convert(condition).sympy_object} : {expression}")
                            return self._normalize(expression, context)
                        elif context.is_known_to_imply(~condition):
                            print(f"normalize eliminate: {SymPyExpression.convert(condition).sympy_object}")
                            pass
                        else:
                            new_arguments.append(self._normalize(expression, context & condition))
                            new_arguments.append(condition)
                    return make_piecewise(new_arguments)
            case FunctionApplication(function=Constant(value=functions.conditional),
                                     arguments=[condition, then, else_]):
                if context.is_known_to_imply(condition):
                    return self._normalize(then, context)
                elif context.is_known_to_imply(~condition):
                    return self._normalize(else_, context)
                else:
                    return if_then_else(condition,
                                        self._normalize(then, context & condition),
                                        self._normalize(else_, context & ~condition))
            case Expression(type=type_) if type_ == bool:
                if context.is_known_to_imply(expression):
                    return basic_true
                if context.is_known_to_imply(~expression):
                    return basic_false
                return if_then_else(expression, True, False)
            case Variable():
                return expression
            case FunctionApplication(is_polynomial=True):  # if function is polynomials, we don't have to normalize: it's integrable
                return expression
            case FunctionApplication(function=function, arguments=arguments):
                with self.profiler.profile_section("function-normalization"):
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
                    with self.profiler.profile_section("quantifier-body-normalization"):
                        normalized_body = self._normalize(body, context & constraint)
                match normalized_body:
                    case FunctionApplication(function=Constant(value=sympy.Piecewise),
                                             arguments=arguments):
                        if arguments[1].contains(index):
                            elements = []
                            for expression, condition in distinct_pairwise(arguments):
                                assert condition.contains(index)
                                with self.profiler.profile_section("quantifier-normalization-after-body"):
                                    elements.append(self._normalize(
                                        BasicQuantifierExpression(operation, index, constraint & condition, expression,
                                                                  is_integral), context, body_is_normalized=True))
                            with self.profiler.profile_section("symbolic addition"):
                                # result = SymPyExpression.new_function_application(operation, elements)
                                result = BasicExpression.new_function_application(operation, elements)
                            return result
                        else:
                            new_arguments = []
                            for expression, condition in distinct_pairwise(arguments):
                                assert not condition.contains(index)
                                new_arguments.append(self._normalize(BasicQuantifierExpression(operation, index, constraint, expression, is_integral), context & condition, body_is_normalized=True))
                                new_arguments.append(condition)
                            with self.profiler.profile_section("make piecewise"):
                                return make_piecewise(new_arguments)

                    case FunctionApplication(function=Constant(value=functions.conditional),
                                             arguments=[condition, then, else_]):
                        # assuming condition is conjunctive (does not contain OR)
                        # if A & B then T {context: A&B} else E {context: ~A | ~B} (bad, we have OR here)
                        #   [2,y]                                     x > y or x < 2
                        # =>
                        # if A then (if B then T {A&B} else E {A&~B}) else (if B then E {~A&B} else E{~A&~B})
                        #                        [2,y]        x<=y and x < 2           x>y and x>=2     x>y and x<=2
                        literals = _split_into_literals(condition)
                        # then and else should have been normalized
                        if len(literals) > 1:
                            print(f"having {len(literals)} literals")
                        return self._normalize_quantifier_expression_given_literals(operation, index, constraint,
                                                                                    is_integral, literals, then, else_,
                                                                                    context)
                    case _:
                        with self.profiler.profile_section("quantifier-normalization-eliminate"):
                            return self._eliminate(operation, index, constraint, normalized_body, is_integral, context)

    def _normalize_function_application(self, function: Expression,
                                        arguments: List[Expression],
                                        context: Z3SolverExpression, i: int = 0) -> Expression:
        if i >= len(arguments):
            return function(*arguments)
            # return _simplifier.simplify(function(*arguments), context)
        arguments[i] = self._normalize(arguments[i], context)
        assert arguments[i] is not None
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
                new_piecewise_arguments = []
                for expression, condition in distinct_pairwise(piecewise_arguments):
                    if context.is_known_to_imply(condition):
                        print(
                            f"move_down shortcut: {SymPyExpression.convert(context).sympy_object} -> "
                            f"{SymPyExpression.convert(condition).sympy_object}")
                        arguments[i] = expression
                        return self._move_down_and_normalize(function, arguments, context & condition, i)
                    elif context.is_known_to_imply(~condition):
                        print(f"move_down eliminate: {SymPyExpression.convert(condition).sympy_object}")
                        pass
                    else:
                        new_arguments = arguments[:]
                        new_arguments[i] = expression
                        new_piecewise_arguments.append(self._move_down_and_normalize(function, new_arguments, context & condition, i))
                        new_piecewise_arguments.append(condition)
                return make_piecewise(new_piecewise_arguments)
            case FunctionApplication(function=Constant(value=functions.conditional),
                                     arguments=[condition, then, else_]):
                then_arguments = arguments[:]
                then_arguments[i] = then
                else_arguments = arguments[:]
                else_arguments[i] = else_

                if context.is_known_to_imply(condition):
                    return self._move_down_and_normalize(function, then_arguments, context & condition, i)
                elif context.is_known_to_imply(~condition):
                    return self._move_down_and_normalize(function, else_arguments, context & ~condition, i)
                else:
                    return if_then_else(condition,
                                        self._move_down_and_normalize(function, then_arguments, context & condition, i),
                                        self._move_down_and_normalize(function, else_arguments, context & ~condition,
                                                                      i))
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
        if isinstance(body, FunctionApplication) and isinstance(body.function, Constant) and \
                body.function.value == functions.conditional:
            raise AttributeError("WHAT")
        if context.is_known_to_imply(~constraint):
            return operation.identity
        result = self._eliminator.eliminate(operation, index, constraint, body, is_integral, context)
        return result  # do not call _simplifier.simplify(result, context), which is quite expensive

    def _normalize_quantifier_expression_given_literals(self,
                                                        operation: AbelianOperation, index: Variable,
                                                        conjunctive_constraint: Context, is_integral: bool,
                                                        literals: List[Expression],
                                                        then: Expression, else_: Expression,
                                                        context: Z3SolverExpression):
        """
        then and else_ are normalized
        when condition is Empty, return
        normalize operation{index:constraint, is_integral}(if literals then then else else_) given context
        in the else, literals are adjusted into "conjunctive form", a & b & c..
        """
        if len(literals) == 1:
            condition = literals[0]
            if condition.contains(index):
                return operation(self._normalize(
                    BasicQuantifierExpression(operation, index, conjunctive_constraint & condition, then, is_integral),
                    context, body_is_normalized=True),
                    self._normalize(
                        BasicQuantifierExpression(operation, index, conjunctive_constraint & ~condition,
                                                  else_, is_integral), context, body_is_normalized=True))
            else:
                if context.is_known_to_imply(condition):
                    return self._normalize(
                        BasicQuantifierExpression(operation, index, conjunctive_constraint, then, is_integral), context,
                        body_is_normalized=True)
                elif context.is_known_to_imply(~condition):
                    return self._normalize(
                        BasicQuantifierExpression(operation, index, conjunctive_constraint, else_, is_integral),
                        context, body_is_normalized=True)
                else:
                    return if_then_else(condition,
                                        self._normalize(
                                            BasicQuantifierExpression(operation, index, conjunctive_constraint, then,
                                                                      is_integral), context, body_is_normalized=True),
                                        self._normalize(
                                            BasicQuantifierExpression(operation, index, conjunctive_constraint, else_,
                                                                      is_integral), context, body_is_normalized=True),
                                        )
        else:
            condition = literals[0]
            if condition.contains(index):
                return operation(self._normalize_quantifier_expression_given_literals(
                    operation, index,
                    conjunctive_constraint & condition,
                    is_integral, literals[1:], then,
                    else_, context),
                    self._normalize(
                        BasicQuantifierExpression(operation, index, conjunctive_constraint & ~condition,
                                                  else_, is_integral), context, body_is_normalized=True))
                # return SymPyExpression.collect(
                #     operation(self._normalize_quantifier_expression_given_literals(operation, index,
                #                                                                    conjunctive_constraint & condition,
                #                                                                    is_integral, literals[1:], then,
                #                                                                    else_, context),
                #               self._normalize(
                #                   BasicQuantifierExpression(operation, index, conjunctive_constraint & ~condition,
                #                                             else_, is_integral), context, body_is_normalized=True)),
                #     index)
            else:
                if context.is_known_to_imply(condition):
                    return self._normalize_quantifier_expression_given_literals(operation, index,
                                                                                conjunctive_constraint, is_integral,
                                                                                literals[1:], then, else_,
                                                                                context & condition)
                elif context.is_known_to_imply(~condition):
                    return self._normalize(
                        BasicQuantifierExpression(operation, index, conjunctive_constraint, else_, is_integral),
                        context, body_is_normalized=True)
                else:
                    return if_then_else(condition,
                                        self._normalize_quantifier_expression_given_literals(operation, index,
                                                                                             conjunctive_constraint,
                                                                                             is_integral, literals[1:],
                                                                                             then, else_,
                                                                                             context & condition),
                                        self._normalize(
                                            BasicQuantifierExpression(operation, index, conjunctive_constraint, else_,
                                                                      is_integral), context, body_is_normalized=True),
                                        )
