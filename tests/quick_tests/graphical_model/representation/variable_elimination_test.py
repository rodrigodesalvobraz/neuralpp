import random
import sys

import torch
from neuralpp.inference.graphical_model.brute_force import BruteForce
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.random.random_model import (
    generate_model,
)
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.graphical_model.variable_elimination import VariableElimination


print("sys.path:", sys.path)


def run_test_ve_against_brute_force(query, model):
    for (i, factor) in enumerate(model):
        print(f"Factor {i}", model[i])
    ve_result = VariableElimination().run(query, model)
    print(f"\nVE marginal on {query}:", ve_result)
    brute_result = BruteForce().run(query, model)
    print(f"\nBrute force marginal on {query}:", brute_result)
    assert ve_result == brute_result


def test_variable_elimination():

    for log_space in {False, True}:

        x = IntegerVariable("x", 3)
        y = IntegerVariable("y", 2)
        z = IntegerVariable("z", 2)

        # noinspection PyShadowingNames
        model = [
            PyTorchTableFactor(
                [], torch.ones(()), log_space=log_space
            ),  # just to test the edge case
            PyTorchTableFactor([x], [0.4, 0.1, 0.5], log_space=log_space),
            PyTorchTableFactor.from_function(
                [y, x],
                lambda y, x: 0.5 if x == 2 else 1.0 if y == x else 0.0,
                log_space=log_space,
            ),
            PyTorchTableFactor.from_function(
                [z, y], lambda z, y: 1.0 if z == y else 0.0, log_space=log_space
            ),
        ]

        query = z

        run_test_ve_against_brute_force(query, model)

    for i in range(10):
        model = generate_model(
            number_of_factors=15, number_of_variables=8, cardinality=2
        )
        query = random.choice([v for f in model for v in f.variables])
        print()
        run_test_ve_against_brute_force(query, model)
