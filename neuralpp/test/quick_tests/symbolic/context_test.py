import pytest
import z3
import sympy

from neuralpp.symbolic.z3_expression import Z3SolverExpression, extract_key_value_from_assertions
from .interpreter_test import dict_to_sympy_context
from z3 import Solver, Ints


def test_context():
    s1 = Solver()
    x, y, z = Ints("x y z")
    s1.add(x == y)
    s1.add(y == 2)
    assert z3.is_eq(y == 2)
    assert not z3.is_var(y)  # z3.is_var means vars in quantifier
    assert not z3.is_var((y == 2).arg(0))
    assert z3.is_int_value((y == 2).arg(1))
    assert extract_key_value_from_assertions([y == 2]) == {'y': 2}
    context1 = Z3SolverExpression(s1)
    with pytest.raises(KeyError):
        context1.dict[y]
    assert context1.dict['y'] == 2
    assert len(context1.dict) == 1
    assert not context1.unsatisfiable

    s2 = Solver()
    s2.add(z == 3)
    context2 = Z3SolverExpression(s2)
    assert not context2.unsatisfiable

    s3 = Solver()
    s3.add(z < x)
    context3 = Z3SolverExpression(s3)
    assert not context3.unsatisfiable

    context4 = context1 & context2
    assert not context4.unsatisfiable
    assert context4.dict['z'] == 3

    context5 = context1 & context2 & context3
    assert context5.unsatisfiable
    with pytest.raises(KeyError):
        # because it's already unsatisfiable, asking for value of a variable does not make sense, since
        # we can return anything
        context5.dict['y']

    context6 = context2 & (z < 0)  # and a literal
    assert context6.unsatisfiable
    context7 = (z < 0) & context2  # rand also works
    assert context7.unsatisfiable

    context8 = context1 & dict_to_sympy_context({'z': 4})  # internally convert other types of expressions
    assert context8.dict['z'] == 4

    s4 = Solver()
    s4.add(x == y)
    s4.add(y == 2)
    s4.add(x == 1)
    with pytest.raises(ValueError):
        Z3SolverExpression(s4)  # because s4 is unsat

