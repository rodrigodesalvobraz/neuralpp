from copy import copy
from typing import Any, Callable, Dict, FrozenSet, List

from z3 import ExprRef, FuncDeclRef, Solver, substitute, Z3_OP_UNINTERPRETED
import z3


z3_false = z3.BoolVal(False)


def z3_merge_solvers(solver1: Solver, solver2: Solver) -> Solver:
    """Make a new solver with constraints from both solver1 and solver2. This function does not modify arguments."""
    result = copy(solver1)
    result.append(solver2.assertions())
    return result


def z3_add_solver_and_literal(solver: Solver, constraint: Any) -> Solver:
    """Make a new solver from the older solver and the added constraint. This function does not modify arguments."""
    result = copy(solver)
    result.add(constraint)
    return result


def z3_replace_in_solver(solver: Solver, from_: ExprRef, to: ExprRef) -> Solver:
    """Make a new solver from the older solver by replacing `from_` to `to`."""
    result = Solver()
    for assertion in solver.assertions():
        result.add(substitute(assertion, (from_, to)))
    return result


def extract_key_to_value_from_assertions(
    assertions: List[z3.ExprRef] | z3.AstVector,
) -> Dict[str, Any]:
    """
    Extract all key-value pairs if it's in the form of "k1 == v1 & k2 == v2 & ..". `And` can be nested:
    we can extract all 4 pairs in And(k1 == v1, k2 == v2, And(k3 == v3, k4 == v4)).
    """

    def key_to_value_accumulator(lhs: z3.ExprRef, rhs: z3.ExprRef):
        if is_z3_variable(lhs) and is_z3_value(rhs):
            result[str(lhs)] = rhs
        elif is_z3_variable(rhs) and is_z3_value(lhs):
            result[str(rhs)] = lhs

    result = {}
    traverse_equalities(assertions, key_to_value_accumulator)
    return result


def traverse_equalities(
    assertions: List[z3.ExprRef],
    accumulator: Callable[[z3.ExprRef, z3.ExprRef], type(None)],
):
    """
    A high-order function that takes a function accumulator as an argument.
    Traverse all the equalities that are leaves of conjunctions and apply accumulator on each of them.
    The accumulator takes the both sides of the equalities as two of its arguments (with lhs comes first).
    """
    for assertion in assertions:
        if z3.is_eq(assertion):
            accumulator(assertion.arg(0), assertion.arg(1))
        elif z3.is_and(assertion):
            traverse_equalities(assertion.children(), accumulator)


def is_z3_uninterpreted_function(declaration: FuncDeclRef) -> bool:
    return declaration.kind() == Z3_OP_UNINTERPRETED


def is_z3_value(v):
    return (
        z3.is_int_value(v)
        or z3.is_rational_value(v)
        or z3.is_algebraic_value(v)
        or z3.is_fp_value(v)
    )


def is_z3_variable(k):
    return z3.is_const(k) and not is_z3_value(k)
