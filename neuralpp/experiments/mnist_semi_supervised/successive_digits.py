import torch

from neuralpp.experiments.mnist_semi_supervised.mnist_semi_supervised import default_parameters, \
    solve_learning_problem_from_parameters, make_digits_and_all_true_constraints_values_batches_generator


# Trains a digit recognizer with the generative model:
#
# Digit[0] ~ Uniform(0..8)
# Digit[1] = Digit[0] + 1
# Image[i] ~ Image_generation(Digit[i]), for i in {0, 1}
#
# dataset: pairs of (Image[0], Image[0])   (no digit labels).
#
# To use negative examples as well, we use a more general model:
#
# # tuples of n digits
# Digit[i] ~ uniform(0..9), for i in 0..(n-1)
# Constraint_i = Digit[i + 1] == Digit[i] + 1
# Image[i] ~ Image_generation(digit[i]), for i in {0, 1}
#
# with dataset in which each example is (Image[0..n-1], Constraint[0..n-1])
#
# It turns out that, during inference, we actually only need
# the *inverse* of image_generation,
# which is provided by a digit recognizer ConvNet.
#
# The corresponding factor graph is:
#
#                                 Constraint0                                   Constraint1
#                                      |                                             |
#                           +---------------------+                       +---------------------+
#                           |   Constraint0 <=>   |                       |   Constraint1 <=>   |
#             Digit0  ------| Digit1 = Digit0 + 1 |------- Digit1 --------| Digit2 = Digit1 + 1 |------- ...
#               |           +---------------------+          |            +---------------------+
#               |                                            |
#      +------------------+                         +------------------+
#      | Digit recognizer |                         | Digit recognizer |
#      +------------------+                         +------------------+
#               |                                            |
#               |                                            |
#             Image0                                       Image1
#
# that is to say, (3n - 1) variables
# Constraint_0, ..., Constraint_(n-2), Digit0, ..., Digit_(n-1), Image0, ... Image_(n-1),
# where Digit_i is constrained to be Digit_(i-1) + 1 iff Constraint_(i-1) is true,
# and Digit_i is the recognition of Image_i.
#
# The original, harder and more interesting case is for n = 2.
#
# We anticipate that presenting pairs images of consecutive digits to the model,
# querying Constraint and minimizing its epoch_average_loss in comparison to the expected value "true"
# will train the recognizer (a shared neural net applied to both images)
# to recognize MNIST digits.
#
# The idea is that images for 0 will only appear in Image0 (because it has no antecessor),
# and images for 9 only in Image_(n-1) (because it has no successor),
# and this will be a strong enough signal for the recognizer to learn those images.
# Once that happens, those images work as the signal for 1 and 8, which work as signals for 2 and 7
# and so on, until all digits are learned.
#
# There are multiple ways of running this experiment. See mnist_semi_supervised.py for extra options.


def indices_of_digit_arguments_of_constraint(constraint_index):
    i = constraint_index
    return i, i + 1


def constraint_function(constraint_index, di, di_plus_one):
    # all constraints use the same predicate, so constraint_index is irrelevant.
    return di_plus_one == di + 1


def generate_chain_of_successive_digits_batch(number_of_digits, chain_length, batch_size):
    """
    Must generate an example guaranteed to satisfy all constraints
    """
    digit_values = []
    last_digit = number_of_digits - 1
    number_of_transitions_in_the_chain = chain_length - 1
    max_value_of_first_digit = last_digit - number_of_transitions_in_the_chain
    exclusive_upper_bound_for_first_digit = max_value_of_first_digit + 1
    digit_values.append(
        torch.randint(exclusive_upper_bound_for_first_digit, (batch_size,))
    )
    for i in range(1, chain_length):
        digit_values.append(digit_values[i - 1] + 1)
    return digit_values


chain_of_successive_digits_and_all_true_constraints_batch_generator = \
    make_digits_and_all_true_constraints_values_batches_generator(
        generate_chain_of_successive_digits_batch
    )

parameters = default_parameters()
parameters.chain_length = 4
parameters.number_of_constraints = parameters.chain_length - 1
parameters.number_of_constraint_values = 2  # constraints are boolean
parameters.indices_of_digit_arguments_of_constraint = indices_of_digit_arguments_of_constraint
parameters.constraint_function = constraint_function

parameters.custom_digits_and_constraints_values_batches_generator = None  # allows negative examples (false constraints)
# parameters.custom_digits_and_constraints_values_batches_generator = chain_of_successive_digits_and_all_true_constraints_batch_generator

solve_learning_problem_from_parameters(parameters)
