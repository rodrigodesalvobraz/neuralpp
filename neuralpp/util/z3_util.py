from z3 import Solver, FuncDeclRef, Z3_OP_UNINTERPRETED
from typing import Any
from copy import copy


def z3_merge_solvers(solver1: Solver, solver2: Solver) -> Solver:
    """  Make a new solver with constraints from both solver1 and solver2. This function does not modify arguments. """
    result = copy(solver1)
    result.append(solver2.assertions())
    return result


def z3_add_solver_and_literal(solver: Solver, constraint: Any) -> Solver:
    """  Make a new solver from the older solver and the added constraint. This function does not modify arguments. """
    result = copy(solver)
    result.add(constraint)
    return result


def is_z3_uninterpreted_function(declaration: FuncDeclRef) -> bool:
    return declaration.kind() == Z3_OP_UNINTERPRETED
