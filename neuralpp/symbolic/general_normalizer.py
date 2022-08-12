from .normalizer import Normalizer
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


_simplifier = ContextSimplifier()
_eliminator = Eliminator()


def conditional_given_context(condition: Expression, then: Expression, else_: Expression, context: Z3SolverExpression):
    """
    """
    if context.is_known_to_imply(condition):
        return then
    elif context.is_known_to_imply(~condition):
        return else_
    else:
        return if_then_else(condition, then, else_,)


def _normalize_conditional(condition: Expression, then: Expression, else_: Expression, context: Z3SolverExpression):
    """
    Arguments condition-then-else_ forms an if-then-else statement.
    If either branch can be pruned (its negation implied by context), it is pruned.
    It is not just an optimization but a requirement that every prunable be pruned:
    otherwise we'd call normalize() with an unsatisfiable context, which is not well-defined.
    """
    if context.is_known_to_imply(condition):
        return _normalize(then, context)
    elif context.is_known_to_imply(~condition):
        return _normalize(else_, context)
    else:
        return if_then_else(condition,
                            _normalize(then, context & condition),
                            _normalize(else_, context & ~condition))


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


def _normalize_quantifier_expression_given_literals(operation: AbelianOperation, index: Variable,
                                                    conjunctive_constraint: Context, is_integral: bool,
                                                    literals: List[Expression],
                                                    then: Expression, else_: Expression,
                                                    context: Z3SolverExpression):
    """
    then and else_ are normalized
    when conditions is Empty, return
    normalize operation{index:constraint, is_integral}(if literals then then else else_) given context
    in the else, literals are adjusted into "conjunctive form", a & b & c..
    """
    if len(literals) == 1:
        condition = literals[0]
        if condition.contains(index):
            return operation(_normalize(BasicQuantifierExpression(operation, index, conjunctive_constraint & condition, then, is_integral), context, body_is_normalized=True),
                             _normalize(BasicQuantifierExpression(operation, index, conjunctive_constraint & ~condition, else_, is_integral), context, body_is_normalized=True))
        else:
            if context.is_known_to_imply(condition):
                return _normalize(BasicQuantifierExpression(operation, index, conjunctive_constraint, then, is_integral), context, body_is_normalized=True)
            elif context.is_known_to_imply(~condition):
                return _normalize(BasicQuantifierExpression(operation, index, conjunctive_constraint, else_, is_integral), context, body_is_normalized=True)
            else:
                return if_then_else(condition,
                                    _normalize(BasicQuantifierExpression(operation, index, conjunctive_constraint, then, is_integral), context, body_is_normalized=True),
                                    _normalize(BasicQuantifierExpression(operation, index, conjunctive_constraint, else_, is_integral), context, body_is_normalized=True),
                                    )
    else:
        condition = literals[0]
        if condition.contains(index):
            return operation(_normalize_quantifier_expression_given_literals(operation, index, conjunctive_constraint & condition, is_integral, literals[1:], then, else_, context),
                             _normalize_quantifier_expression_given_literals(operation, index, conjunctive_constraint & ~condition, is_integral, literals[1:], else_, else_, context))
        else:
            if context.is_known_to_imply(condition):
                return _normalize_quantifier_expression_given_literals(operation, index, conjunctive_constraint, is_integral, literals[1:], then, else_, context & condition)
            elif context.is_known_to_imply(~condition):
                return _normalize_quantifier_expression_given_literals(operation, index, conjunctive_constraint, is_integral, literals[1:], else_, else_, context & ~condition)
            else:
                return if_then_else(condition,
                                    _normalize_quantifier_expression_given_literals(operation, index, conjunctive_constraint, is_integral, literals[1:], then, else_, context & condition),
                                    _normalize_quantifier_expression_given_literals(operation, index, conjunctive_constraint, is_integral, literals[1:], else_, else_, context & ~condition),
                                    )


def _normalize(expression: Expression, context: Z3SolverExpression, body_is_normalized: bool = False) -> Expression:
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
        case FunctionApplication(function=Constant(value=functions.conditional),
                                 arguments=[condition, then, else_]):
            if context.is_known_to_imply(condition):
                return _normalize(then, context)
            elif context.is_known_to_imply(~condition):
                return _normalize(else_, context)
            else:
                return if_then_else(condition,
                                    _normalize(then, context & condition),
                                    _normalize(else_, context & ~condition))
        case Expression(type=type_) if type_ == bool:
            if context.is_known_to_imply(expression):
                return basic_true
            if context.is_known_to_imply(~expression):
                return basic_false
            return if_then_else(expression, True, False)
        case Variable():
            return expression
        case FunctionApplication(function=function, arguments=arguments):
            return _normalize_function_application(function, arguments, context)
        case QuantifierExpression(operation=operation, index=index, constraint=constraint, body=body,
                                  is_integral=is_integral):
            if context.contains(index):
                raise ValueError(f"context {context} should not contain index {index}")
            if context.is_known_to_imply(~constraint):
                return operation.identity

            if body_is_normalized:
                normalized_body = body
            else:
                normalized_body = _normalize(body, context & constraint)

            match normalized_body:
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
                    return _normalize_quantifier_expression_given_literals(operation, index, constraint, is_integral,
                                                                           literals, then, else_, context)
                    if condition.contains(index):
                        return _normalize(
                            operation(BasicQuantifierExpression(operation, index, constraint & condition, then,
                                                                is_integral),
                                      BasicQuantifierExpression(operation, index, constraint & ~condition, else_,
                                                                is_integral)),
                            context)
                    else:
                        return _normalize_conditional(condition,
                                                      BasicQuantifierExpression(operation, index, constraint, then,
                                                                                is_integral),
                                                      BasicQuantifierExpression(operation, index, constraint, else_,
                                                                                is_integral),
                                                      context)
                case _:
                    return _eliminate(operation, index, constraint, normalized_body, is_integral, context)


def _eliminate(operation: AbelianOperation, index: Variable, constraint: Context, body: Expression,
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
    print(f"eliminating: {body}")
    result = _eliminator.eliminate(operation, index, constraint, body, is_integral, context)
    print(f"done")
    return _simplifier.simplify(result, context)


def _normalize_function_application(function: Expression,
                                    arguments: List[Expression],
                                    context: Z3SolverExpression, i: int = 0) -> Expression:
    if i >= len(arguments):
        return function(*arguments)
        # return _simplifier.simplify(function(*arguments), context)
    arguments[i] = _normalize(arguments[i], context)
    return _move_down_and_normalize(function, arguments, context, i)


def _move_down_and_normalize(function: Expression,
                             arguments: List[Expression],
                             context: Z3SolverExpression, i: int) -> Expression:
    """
    assume i < len(arguments);
    assume all {arguments[j] | j < i} does not contain if-then-else (and thus normalized);
    assume arguments[i] has been normalized.
    move down the f to the leaves. Then replace every leaf f with its normalized version
    """
    match arguments[i]:
        case FunctionApplication(function=Constant(value=functions.conditional),
                                 arguments=[condition, then, else_]):
            then_arguments = arguments[:]
            then_arguments[i] = then
            else_arguments = arguments[:]
            else_arguments[i] = else_

            if context.is_known_to_imply(condition):
                return _move_down_and_normalize(function, then_arguments, context & condition, i)
            elif context.is_known_to_imply(~condition):
                return _move_down_and_normalize(function, else_arguments, context & ~condition, i)
            else:
                return if_then_else(condition,
                                    _move_down_and_normalize(function, then_arguments, context & condition, i),
                                    _move_down_and_normalize(function, else_arguments, context & ~condition, i))
        case _:
            # _check_no_conditional(arguments[i])
            return _normalize_function_application(function, arguments, context, i + 1)


def _check_no_conditional(argument: Expression):
    match argument:
        case FunctionApplication(function=Constant(value=functions.conditional)):
            raise AttributeError("WRONG")
        case _:
            for subexpression in argument.subexpressions:
                _check_no_conditional(subexpression)


class GeneralNormalizer(Normalizer):
    @staticmethod
    def normalize(expression: Expression, context: Z3SolverExpression) -> Expression:
        with sympy_evaluate(True):
            return _normalize(expression, context)
