from typing import Any, Dict

from neuralpp.inference.graphical_model.representation.factor.neural.neural_factor import (
    NeuralFactor,
)
from neuralpp.inference.graphical_model.variable.discrete_variable import (
    DiscreteVariable,
)
from neuralpp.util import util
from neuralpp.util.util import merge_dicts


def check_and_show_conditional_distributions(
    neural_factor, ground_truth=None, tolerance=0.05
):
    """
    Checks if neural_factor implements a distribution (probabilities summing up to 1 for each input assignment)
    and shows it in the console.
    If a ground truth function is provided, checks the neural factor distribution against it.
    The ground truth is passed an assignment as an unpacked list in the same order as the variables in the factor.
    """
    input_variables = neural_factor.input_variables
    output_variable = neural_factor.output_variable
    for input_assignment_dict in DiscreteVariable.assignments_product_dicts(
        input_variables
    ):

        potentials = [
            compute_potential(
                neural_factor, input_assignment_dict, output_assignment_dict
            )
            for output_assignment_dict in DiscreteVariable.assignments_product_dicts(
                [output_variable]
            )
        ]
        assert abs(sum(potentials) - 1) < 0.0001

        for output_assignment_dict in DiscreteVariable.assignments_product_dicts(
            [output_variable]
        ):
            potential, assignment_dict = compute_potential_and_assignment(
                neural_factor, input_assignment_dict, output_assignment_dict
            )
            print("----------------------")
            print(
                f"Neural factor potential for {assignment_dict}:",
                util.rounded_list(potential),
                end="",
            )

            if ground_truth is not None:
                assignment = [assignment_dict[v] for v in neural_factor.variables]
                ground_probability = ground_truth(*assignment)
                print(f" (ground truth: {round(ground_probability, 3)})")
                if not util.close(potential, ground_probability, tolerance):
                    raise AssertionError(
                        f"Probability {round(potential.item(), 3)} not close enough to "
                        f"ground truth {ground_probability} (tolerance is {tolerance})"
                    )
            else:
                print()


def compute_potential(
    neural_factor: NeuralFactor,
    input_assignment_dict: Dict[DiscreteVariable, Any],
    output_assignment_dict: Dict[DiscreteVariable, Any],
):
    potential, assignment_dict = compute_potential_and_assignment(
        neural_factor, input_assignment_dict, output_assignment_dict
    )
    return potential


def compute_potential_and_assignment(
    neural_factor: NeuralFactor,
    input_assignment_dict: Dict[DiscreteVariable, Any],
    output_assignment_dict: Dict[DiscreteVariable, Any],
):
    assignment_dict = merge_dicts(input_assignment_dict, output_assignment_dict)
    potential = neural_factor(assignment_dict)
    return potential, assignment_dict
