import random

import pytest

from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import PyTorchTableFactor
from neuralpp.inference.graphical_model.representation.factor.switch_factor import SwitchFactor
from neuralpp.inference.graphical_model.representation.random.multi_type_random_model import MultiTypeRandomModel, \
    FactorMaker
from neuralpp.inference.graphical_model.representation.random.multi_type_random_model_util \
    import make_standard_gaussian, make_gaussian_with_mean, make_switch_of_gaussians_with_mean
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.graphical_model.variable.tensor_variable import TensorVariable
from neuralpp.util.util import print_dict_in_lines


def test_single_seed_variable():
    random_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary=1,
            from_type_to_number_of_seed_variables={
                TensorVariable: 1,
            }, factor_makers=[
                FactorMaker([TensorVariable],
                            make_standard_gaussian)
            ], from_type_to_variable_maker={})

    x1 = TensorVariable("x1")
    expected = {x1: make_standard_gaussian(x1)}

    print("Expected:")
    print_dict_in_lines(expected)

    print("Actual:")
    print_dict_in_lines(random_model.from_variable_to_distribution)

    assert random_model.from_variable_to_distribution == expected


def test_multiple_seed_variables():
    random_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary=1,
            from_type_to_number_of_seed_variables={
                TensorVariable: 3,
            }, factor_makers=[
                FactorMaker([TensorVariable],
                            make_standard_gaussian)
            ], from_type_to_variable_maker={})

    x1 = TensorVariable("x1")
    x2 = TensorVariable("x2")
    x3 = TensorVariable("x3")
    expected = {
        x1: make_standard_gaussian(x1),
        x2: make_standard_gaussian(x2),
        x3: make_standard_gaussian(x3),
    }

    print("Expected:")
    print_dict_in_lines(expected)

    print("Actual:")
    print_dict_in_lines(random_model.from_variable_to_distribution)

    assert random_model.from_variable_to_distribution == expected


def test_zero_seed_variables():
    random_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary=1,
            from_type_to_number_of_seed_variables={
                TensorVariable: 0,  # ZERO seed variables
            }, factor_makers=[
                FactorMaker([TensorVariable],
                            make_standard_gaussian)
            ], from_type_to_variable_maker={})

    expected = {}

    print("Expected:")
    print_dict_in_lines(expected)

    print("Actual:")
    print_dict_in_lines(random_model.from_variable_to_distribution)

    assert random_model.from_variable_to_distribution == expected


def test_depth_two_tree():
    random_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary=6,
            from_type_to_number_of_seed_variables={
                TensorVariable: 3,
            },
            factor_makers=[
                FactorMaker([TensorVariable],
                            make_standard_gaussian),
                FactorMaker([TensorVariable, TensorVariable],
                            make_gaussian_with_mean),
            ],
            loop_coefficient=0.0)

    # Graph is formed in a breadth-first way, so x1, x2, x3 are the seed values,
    # and have mean parents (haha, see what I did there?) x4, x5, x6 respectively.

    expected = {}
    for i in range(1, 4):
        seed_variable = TensorVariable(f"x{i}")
        mean_parent = TensorVariable(f"x{i + 3}")
        expected[seed_variable] = make_gaussian_with_mean([seed_variable, mean_parent])
    for i in range(4, 7):
        mean_parent = TensorVariable(f"x{i}")
        expected[mean_parent] = make_standard_gaussian(mean_parent)

    print("Expected:")
    print_dict_in_lines(expected)

    print("Actual:")
    print_dict_in_lines(random_model.from_variable_to_distribution)

    assert random_model.from_variable_to_distribution == expected


def test_depth_two_attempted_loops():
    random_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary=6,
            from_type_to_number_of_seed_variables={
                TensorVariable: 3,
            },
            factor_makers=[
                FactorMaker([TensorVariable],
                            make_standard_gaussian),
                FactorMaker([TensorVariable, TensorVariable],
                            make_gaussian_with_mean),
            ],
            loop_coefficient=1.0)

    x1 = TensorVariable("x1")
    x2 = TensorVariable("x2")
    x3 = TensorVariable("x3")
    x4 = TensorVariable("x4")
    x5 = TensorVariable("x5")
    x6 = TensorVariable("x6")

    # We will try to reuse one of the three seed variables x1, x2, x3 for their parents.
    # Possibilities tree:
    # x1 picks x2
    #       x2 picks x3 (to avoid a directed cycle)
    #           x3 needs a new parent x4 to avoid a cycle (expected 1)
    # x1 picks x3
    #       x2 picks x1
    #           x3 needs a new parent x4 to avoid a cycle (expected 2)
    #       x2 picks x3
    #           x3 needs a new parent x4 to avoid a cycle (expected 3)

    invariant = {
        x3: make_gaussian_with_mean(x3, x4),
        x4: make_gaussian_with_mean(x4, x5),
        x5: make_gaussian_with_mean(x5, x6),
        x6: make_standard_gaussian(x6),
    }

    expected = [
        {
            x1: make_gaussian_with_mean(x1, x2),
            x2: make_gaussian_with_mean(x2, x3),
            **invariant,
        },
        {
            x1: make_gaussian_with_mean(x1, x3),
            x2: make_gaussian_with_mean(x2, x1),
            **invariant,
        },
        {
            x1: make_gaussian_with_mean(x1, x3),
            x2: make_gaussian_with_mean(x2, x3),
            **invariant,
        },
    ]

    for i, e in enumerate(expected, start=1):
        print(f"Expected {i}:")
        print_dict_in_lines(e)

    print("Actual:")
    print_dict_in_lines(random_model.from_variable_to_distribution)

    assert random_model.from_variable_to_distribution in expected


def test_multi_type():

    random.seed(3)

    random_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary=6,
            from_type_to_number_of_seed_variables={
                IntegerVariable: 1,
                TensorVariable: 3,
            },
            factor_makers=[
                FactorMaker([TensorVariable],
                            make_standard_gaussian),
                FactorMaker([TensorVariable, TensorVariable],
                            make_gaussian_with_mean),
                FactorMaker([TensorVariable, IntegerVariable, TensorVariable, TensorVariable],
                            make_switch_of_gaussians_with_mean),
                FactorMaker([IntegerVariable],
                            lambda variables: PyTorchTableFactor(variables, [0.4, 0.6])),
                FactorMaker([IntegerVariable, IntegerVariable],
                            lambda variables: PyTorchTableFactor(variables, [[0.4, 0.3], [0.6, 0.7]])),
            ],
            from_type_to_variable_maker={IntegerVariable: lambda name: IntegerVariable(name, 2)},
            loop_coefficient=1.0)

    print_dict_in_lines(random_model.from_variable_to_distribution)

    x1 = IntegerVariable("x1", 2)
    x2 = TensorVariable("x2")
    x3 = TensorVariable("x3")
    x4 = TensorVariable("x4")
    x5 = IntegerVariable("x5", 2)
    x6 = TensorVariable("x6")
    x7 = TensorVariable("x7")

    expected = {
        x1: PyTorchTableFactor([x1, x5], [[0.4, 0.3], [0.6, 0.7]]),
        x2: make_gaussian_with_mean(x2, x4),
        x3: SwitchFactor(x1, [make_gaussian_with_mean(x3, x2), make_gaussian_with_mean(x3, x4)]),
        x4: SwitchFactor(x1, [make_gaussian_with_mean(x4, x6), make_gaussian_with_mean(x4, x7)]),
        x5: PyTorchTableFactor([x5], [0.4, 0.6]),
        x6: make_standard_gaussian(x6),
        x7: make_standard_gaussian(x7),
    }

    print("Expected:")
    print_dict_in_lines(expected)

    print("Actual:")
    print_dict_in_lines(random_model.from_variable_to_distribution)

    assert random_model.from_variable_to_distribution == expected


@pytest.fixture(params=[1.0, 0.5, 0.0])
def loop_coefficient(request):
    return request.param


def test_multi_type_large_scale(loop_coefficient):
    random.seed(0)

    multiplier = 100  # multiplier == 1000, loop_coefficient == 1.0 took around 24 s on a MacPro as of March 2022.

    random_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary=6 * multiplier,
            from_type_to_number_of_seed_variables={
                IntegerVariable: 1 * multiplier,
                TensorVariable: 3 * multiplier,
            },
            factor_makers=[
                FactorMaker([TensorVariable],
                            make_standard_gaussian),
                FactorMaker([TensorVariable, TensorVariable],
                            make_gaussian_with_mean),
                FactorMaker([TensorVariable, IntegerVariable, TensorVariable, TensorVariable],
                            make_switch_of_gaussians_with_mean),
                FactorMaker([IntegerVariable],
                            lambda variables: PyTorchTableFactor(variables, [0.4, 0.6])),
                FactorMaker([IntegerVariable, IntegerVariable],
                            lambda variables: PyTorchTableFactor(variables, [[0.4, 0.3], [0.6, 0.7], ])),
            ],
            from_type_to_variable_maker={IntegerVariable: lambda name: IntegerVariable(name, 2)},
            loop_coefficient=loop_coefficient,
        )

    number_of_variables = len(random_model.from_variable_to_distribution)
    print("Number of variables:", number_of_variables)
    print_dict_in_lines(random_model.from_variable_to_distribution)

    assert 6 * multiplier <= number_of_variables <= 8 * multiplier
