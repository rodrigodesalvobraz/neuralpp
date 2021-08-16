from types import SimpleNamespace

import torch

from neuralpp.experiments.mnist_semi_supervised.mnist_semi_supervised import default_parameters, \
    solve_learning_problem_from_parameters
from neuralpp.inference.graphical_model.learn.learning_problem_solver import solve_learning_problem, LearningProblem
from neuralpp.inference.graphical_model.learn.uniform_training import UniformTraining
from neuralpp.inference.graphical_model.representation.factor.fixed.fixed_pytorch_factor import (
    FixedPyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.factor.neural.neural_factor import (
    NeuralFactor,
)
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.table.pytorch_log_table import (
    PyTorchLogTable,
)
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.graphical_model.variable.tensor_variable import TensorVariable
from neuralpp.inference.neural_net.ConvNet import ConvNet
from neuralpp.inference.neural_net.MLP import MLP
from neuralpp.inference.neural_net.from_log_to_probabilities_adapter import (
    FromLogToProbabilitiesAdapter,
)
from neuralpp.util.data_loader_from_random_data_point_thunk import (
    data_loader_from_batch_generator,
)
from neuralpp.util.generic_sgd_learner import default_after_epoch
from neuralpp.util.mnist_util import read_mnist, show_images_and_labels
from neuralpp.util.util import join, set_seed

# Trains a digit recognizer with the following factor graph:
#
#                      Constraint0
#                           |
#        +------------------------------------+
#        |    Constraint0 = Digit0 + Digit1   |
#        +------------------------------------+
#               |                      |
#            Digit0                  Digit1
#               |                      |
#               |                      |
#      +------------------+   +------------------+
#      | Digit recognizer |   | Digit recognizer |
#      +------------------+   +------------------+
#               |                      |
#               |                      |
#             Image0                 Image1


def indices_of_digit_arguments_of_constraint(constraint_index):
    return 0, 1


def constraint_function(constraint_index, d0, d1):
    # Using a single constraint, so constraint_index is irrelevant
    return d0 + d1


def generate_chain_of_summands_digits_values_and_constraint_equal_to_sum(number_of_digits,
        chain_length, # ignored because we have always two digits
        number_of_constraints, # ignored because we have always 1 constraint
        batch_size):
    d0_values = torch.randint(number_of_digits, (batch_size,))
    max_d1_value = number_of_digits - d0_values
    d1_values = torch.cat([torch.randint(max_d1_value[i], (1,)) for i in range(batch_size)])
    constraint_values = d0_values + d1_values
    # print(d0_values[0].item(), d1_values[0].item(), constraint_values[0].item())
    return [d0_values, d1_values], [constraint_values]


parameters = default_parameters()
parameters.chain_length = 2
parameters.number_of_constraints = 1
parameters.number_of_constraint_values = parameters.number_of_digits
parameters.indices_of_digit_arguments_of_constraint = indices_of_digit_arguments_of_constraint
parameters.constraint_function = constraint_function
parameters.number_of_constraints = 1
# parameters.custom_digits_and_constraints_values_batches_generator = None # not available because it generates invalid examples (eg 6 + 7 = 13)
parameters.custom_digits_and_constraints_values_batches_generator = generate_chain_of_summands_digits_values_and_constraint_equal_to_sum

solve_learning_problem_from_parameters(parameters)
