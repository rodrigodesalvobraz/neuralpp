import torch

from neuralpp.inference.graphical_model.representation.factor.continuous.normal_factor import NormalFactor
from neuralpp.inference.graphical_model.representation.random.multi_type_random_model import MultiTypeRandomModel, \
    FactorMaker
from neuralpp.inference.graphical_model.variable.tensor_variable import TensorVariable
from neuralpp.util.util import print_dict_in_lines


def make_standard_normal(variable):
    mu = TensorVariable(f"mu_{{{variable}}}")
    std_dev = TensorVariable(f"std_dev_{{{variable}}}")
    return NormalFactor([variable, mu, std_dev], conditioning_dict={mu: torch.tensor(0.0), std_dev: torch.tensor(1.0)})


def make_normal_with_mean(variable, mu):
    std_dev = TensorVariable(f"std_dev_{{{variable}}}")
    return NormalFactor([variable, mu, std_dev], conditioning_dict={std_dev: torch.tensor(1.0)})


def test_single_seed_variable():
    random_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_generate=1,
            from_type_to_number_of_seed_variables={
                TensorVariable: 1,
            },
            variable_maker=lambda type, name: TensorVariable(name),
            factor_makers=[
                FactorMaker([TensorVariable],
                            lambda variables: make_standard_normal(variables[0]))
            ],
        )

    x1 = TensorVariable("x1")
    expected = {x1: make_standard_normal(x1)}

    print("Expected:")
    print_dict_in_lines(expected)

    print("Actual:")
    print_dict_in_lines(random_model.from_variable_to_distribution)

    assert random_model.from_variable_to_distribution == expected


def test_multiple_seed_variables():
    random_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_generate=1,
            from_type_to_number_of_seed_variables={
                TensorVariable: 3,
            },
            variable_maker=lambda type, name: TensorVariable(name),
            factor_makers=[
                FactorMaker([TensorVariable],
                            lambda variables: make_standard_normal(variables[0]))
            ],
        )

    x1 = TensorVariable("x1")
    x2 = TensorVariable("x2")
    x3 = TensorVariable("x3")
    expected = {
        x1: make_standard_normal(x1),
        x2: make_standard_normal(x2),
        x3: make_standard_normal(x3),
    }

    print("Expected:")
    print_dict_in_lines(expected)

    print("Actual:")
    print_dict_in_lines(random_model.from_variable_to_distribution)

    assert random_model.from_variable_to_distribution == expected


def test_zero_seed_variables():
    random_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_generate=1,
            from_type_to_number_of_seed_variables={
                TensorVariable: 0,  # ZERO seed variables
            },
            variable_maker=lambda type, name: TensorVariable(name),
            factor_makers=[
                FactorMaker([TensorVariable],
                            lambda variables: make_standard_normal(variables[0]))
            ],
        )

    expected = {}

    print("Expected:")
    print_dict_in_lines(expected)

    print("Actual:")
    print_dict_in_lines(random_model.from_variable_to_distribution)

    assert random_model.from_variable_to_distribution == expected


def test_depth_two():
    random_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_generate=6,
            from_type_to_number_of_seed_variables={
                TensorVariable: 3,
            },
            variable_maker=lambda type, name: TensorVariable(name),
            factor_makers=[
                FactorMaker([TensorVariable],
                            lambda variables: make_standard_normal(variables[0])),
                FactorMaker([TensorVariable, TensorVariable],
                            lambda variables: make_normal_with_mean(variables[0], variables[1])),
            ],
            loop_coefficient=0.0  # enforces no loops (all parents are new variables, forming a tree)
        )

    # Graph is formed in a breadth-first way, so x1, x2, x3 are the seed values,
    # and have mean parents (haha, see what I did there?) x4, x5, x6 respectively.

    expected = {}
    for i in range(1, 4):
        seed_variable = TensorVariable(f"x{i}")
        mean_parent = TensorVariable(f"x{i + 3}")
        expected[seed_variable] = make_normal_with_mean(seed_variable, mean_parent)
    for i in range(4, 7):
        mean_parent = TensorVariable(f"x{i}")
        expected[mean_parent] = make_standard_normal(mean_parent)

    print("Expected:")
    print_dict_in_lines(expected)

    print("Actual:")
    print_dict_in_lines(random_model.from_variable_to_distribution)

    assert random_model.from_variable_to_distribution == expected


