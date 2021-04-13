import torch

from inference.graphical_model.learn.learn import default_learning_hook, NeuralPPLearner
from inference.graphical_model.representation.factor.fixed.fixed_pytorch_factor import FixedPyTorchTableFactor
from inference.graphical_model.representation.factor.neural.neural_factor import NeuralFactor
from inference.graphical_model.representation.factor.pytorch_table_factor import PyTorchTableFactor
from inference.graphical_model.representation.table.pytorch_log_table import PyTorchLogTable
from inference.graphical_model.variable.integer_variable import IntegerVariable
from inference.graphical_model.variable.tensor_variable import TensorVariable
from inference.neural_net.ConvNet import ConvNet
from inference.neural_net.MLP import MLP
from inference.neural_net.from_log_to_probabilities_adapter import FromLogToProbabilitiesAdapter
from util.data_loader_from_random_data_point_thunk import data_loader_from_random_data_point_generator

# Trains a digit recognizer with the following factor graph:
#
#                                 Constraint
#                                      |
#                           +---------------------+
#                           |   Constraint <=>    |
#             Digit0  ------| Digit1 = Digit0 + 1 |------- Digit1
#               |           +---------------------+          |
#               |                                            |
#      +------------------+                         +------------------+
#      | Digit recognizer |                         | Digit recognizer |
#      +------------------+                         +------------------+
#               |                                            |
#               |                                            |
#             Image0                                       Image1
#
# that is to say, five variables Constraint, Digit0, Digit1, Image0, Image1,
# where Digit1 is constrained to be Digit0 + 1 iff Constraint is true,
# and Digit_i is the recognition of Image_i.
#
# We anticipate that presenting pairs images of consecutive digits to the model,
# querying Constraint and minimizing its loss in comparison to the expected value "true"
# will train the recognizer (a shared neural net applied to both images)
# to recognize MNIST digits.
#
# The idea is that images for 0 will only appear in Image0, and images for 9 only in Image1,
# and this will be a strong enough signal for the recognizer to learn those images.
# Once that happens, those images work as the signal for 1 and 8, which work as signals for 2 and 7
# and so on, until all digits are learned.
#
# This script offers multiple ways of running the above, including a simplified setting
# in which the "images" are just integers from 0 to 9.
# Even in this very simple setting, the recognizer still needs to learn to associate each digit to itself.
# This provides a way of testing the general dynamic of the model without dealing with the complexities
# of actual image recognition.
#
# Other options include whether the two recognizers are the same network or not
# (essential for learing from positive examples only),
# whether negative examples are present (non-consecutive digits with Constraint = false),
# whether to use a single digit image per digit,
# and various possible initializations for the recognizer.

from util.mnist_util import read_mnist, show_images_and_labels

from util.util import join, set_default_tensor_type_and_return_device

# -------------- PARAMETERS

number_of_digits = 10
use_real_images = False  # use real images; otherwise, use its digit value only as input (simpler version of experiment)
show_examples = True  # show some examples of images (sanity check for data structure)
use_a_single_image_per_digit = False  # to make the problem easier -- removes digit variability from the problem
try_cuda = True
batch_size = 50
epoch_size = 1000
number_of_epochs_between_evaluations = 1
max_real_mnist_datapoints = None
seed = None   # use None for non-deterministic seed

use_positive_examples_only = True  # show only examples of consecutive pairs with the constraint labeled True,
# as opposed to random pairs with constraint labeled True or False accordingly.
# Including negative examples makes the problem easier, but less
# uniform random tables still get stuck in local minima.

use_shared_recognizer = True  # the same recognizer is shared by both "image" -> digit pairs
# Note that, if given positive examples only, we need a shared recognizer for
# learning to have a chance of succeeding
# because otherwise the two recognizers never get to see 9 and 0 respectively.

# 'recognizer' below allows the selection of different functions and initializations for recognizer.
# If use_real_images is True, this selection is ignored and a ConvNet is used, along with number_of_digits = 10.

#recognizer = "neural net"  # if use_real_images, a ConvNet
                            # if not use_real_images, an MLP with a single input unit, number_of_digits hidden units,
                            # and number_of_digits output units.
#recognizer = "fixed ground truth table"  # the correct table, with fixed parameters.
#recognizer = "random table"  # an initialization with random potentials; see parameter below for increasing randomness
#recognizer = "noisy left-shift"  # a hard starting point, in which digits map to their left-shift.
                                   # This will get most consecutive pairs to satisfy the constraint
                                   # but for (0, 1) and (8,9), even though *every* digit is being misclassified.
                                   # See parameter below for selecting noise level.
                                   # Making this noisier helps because it "dillutes" the hard starting point.
recognizer_type = "uniform"  # a uniform table -- learning works well


left_shift_noise = .1  # probability of noisy left-shift recognizer initialization not left-shifting
                       # but hitting some other digit uniformly

upper_bound_for_log_potential_in_random_table = 0.1  # log of potentials are uniformly sampled from
                                                   # [0, upper_bound_for_log_potential_in_random_table].
                                                   # The higher the value, the farther from the uniform the table is.
                                                   # So far we observe that tables farther from the uniform
                                                   # often get stuck in local minima,
                                                   # and that uniform tables always converge to the correct answer.

# -------------- END OF PARAMETERS


# -------------- PROCESSING PARAMETERS
if use_real_images:
    if number_of_digits != 10 or recognizer_type != "neural net":
        print("Using real images; forcing number of digits to 10 and recognizer type to neural net")
    number_of_digits = 10
    recognizer_type = "neural net"

if recognizer_type == "uniform":
    recognizer_type = "random table"
    upper_bound_for_log_potential_in_random_table = 0  # A value of 0 samples from [0, 0], providing a uniform table.

lr = 1e-3 if recognizer_type == "neural net" else 1e-2
# -------------- END OF PROCESSING PARAMETERS


def main():

    set_seed()

    # Create random variables
    global i0, i1, d0, d1, constraint  # so they are easily accessible in later functions
    i0 = TensorVariable("i0") if use_real_images else IntegerVariable("i0", number_of_digits)
    i1 = TensorVariable("i1") if use_real_images else IntegerVariable("i1", number_of_digits)
    d0 = IntegerVariable("d0", number_of_digits)
    d1 = IntegerVariable("d1", number_of_digits)
    constraint = IntegerVariable("constraint", 2)

    # Load images, if needed, before setting default device to cuda
    global from_digit_batch_to_image_batch
    if use_real_images:
        global images_by_digits_by_phase  # so they are easily accessible in later functions
        global next_image_index_by_digit
        images_by_digits_by_phase = read_mnist(max_real_mnist_datapoints)
        next_image_index_by_digit = {d: 0 for d in range(number_of_digits)}
        if show_examples:
            images = [images_by_digits_by_phase["train"][d][i] for i in range(5) for d in range(number_of_digits)]
            labels = [d for i in range(5) for d in range(number_of_digits)]
            show_images_and_labels(5, 10, images, labels)
        from_digit_batch_to_image_batch = get_next_real_image_batch_for_digit_batch
    else:
        from_digit_batch_to_image_batch = get_next_fake_image_batch_for_digit_batch

    train_data_loader = make_data_loader()

    device = set_default_tensor_type_and_return_device(try_cuda)
    print(f"Using {device} device")

    # Creating model after attempting to set default tensor type to cuda so it sits there
    global constraint_factor, i0_d0, i1_d1  # so they are easily accessible in later functions
    constraint_factor = make_constraint_factor()
    i0_d0, i1_d1 = make_recognizer_factors()

    global model
    model = [
        # IMPORTANT: this particular factor order is relied upon later in the code
        constraint_factor,
        i0_d0,
        i1_d1,
    ]

    print("\nInitial model:")
    print(join(model, "\n"))
    print("\nInitial evaluation:")
    print_digit_evaluation()

    if recognizer_type != "fixed ground truth table":
        print("Learning...")
        NeuralPPLearner.learn(model, train_data_loader, device=device, lr=lr, after_epoch=after_epoch)

    print("\nFinal model:")
    print(join(model, "\n"))
    print("\nFinal evaluation:")
    print_digit_evaluation()


def make_constraint_factor():
    constraint_predicate = lambda d0, d1, constraint: int(d1 == d0 + 1) == constraint
    constraint_factor = FixedPyTorchTableFactor.from_predicate((d0, d1, constraint), constraint_predicate)
    return constraint_factor


def make_recognizer_factors():
    if recognizer_type == "neural net":
        if use_real_images:
            neural_net_maker = lambda: FromLogToProbabilitiesAdapter(ConvNet())
        else:
            neural_net_maker = lambda: MLP(1, number_of_digits, number_of_digits)
        neural_net1 = neural_net_maker()
        neural_net2 = neural_net1 if use_shared_recognizer else neural_net_maker()
        i0_d0 = NeuralFactor(neural_net1, input_variables=[i0], output_variable=d0)
        i1_d1 = NeuralFactor(neural_net2, input_variables=[i1], output_variable=d1)

    elif recognizer_type == "fixed ground truth table":
        predicate = lambda i, d: d == i
        image_and_digit_table = PyTorchTableFactor.from_predicate([i0, d0], predicate, log_space=True).table
        i0_d0 = PyTorchTableFactor([i0, d0], image_and_digit_table)
        i1_d1 = PyTorchTableFactor([i1, d1], image_and_digit_table)

    elif recognizer_type == "noisy left-shift":
        probability_of_left_shift = 1 - left_shift_noise
        number_of_non_left_shift = number_of_digits - 1
        probability_of_each_non_left_shift = left_shift_noise / number_of_non_left_shift
        left_shift_pairs = {(i, (i - 1) % number_of_digits) for i in range(number_of_digits)}

        def potential(i, d):
            return probability_of_left_shift if (i, d) in left_shift_pairs else probability_of_each_non_left_shift

        if use_shared_recognizer:
            image_and_digit_table1 = PyTorchTableFactor.from_function([i0, d0], potential, log_space=True).table
            image_and_digit_table2 = image_and_digit_table1
        else:
            image_and_digit_table1 = PyTorchTableFactor.from_function([i0, d0], potential, log_space=True).table
            image_and_digit_table2 = PyTorchTableFactor.from_function([i1, d1], potential, log_space=True).table

        i0_d0 = PyTorchTableFactor([i0, d0], image_and_digit_table1)
        i1_d1 = PyTorchTableFactor([i1, d1], image_and_digit_table2)

    elif recognizer_type == "random table":
        def make_random_parameters():
            return (torch.rand(number_of_digits, number_of_digits)
                    * upper_bound_for_log_potential_in_random_table).requires_grad_(True)

        if use_shared_recognizer:
            image_and_digit_table1 = PyTorchLogTable(make_random_parameters())
            image_and_digit_table2 = image_and_digit_table1
        else:
            image_and_digit_table1 = PyTorchLogTable(make_random_parameters())
            image_and_digit_table2 = PyTorchLogTable(make_random_parameters())
        i0_d0 = PyTorchTableFactor([i0, d0], image_and_digit_table1)
        i1_d1 = PyTorchTableFactor([i1, d1], image_and_digit_table2)

    else:
        raise Exception(f"Unknown recognizer type: {recognizer_type}")

    return i0_d0, i1_d1


def make_data_loader():
    random_pair_generator = \
        random_positive_example_generator() if use_positive_examples_only \
            else random_positive_or_negative_example_generator()
    train_data_loader = data_loader_from_random_data_point_generator(epoch_size, random_pair_generator, print=None)
    return train_data_loader


def get_next_fake_image_batch_for_digit_batch(digit_batch):
    return digit_batch


def get_next_real_image_batch_for_digit_batch(digit_batch):
    images_list = []
    for d in digit_batch:
        d = d.item()
        image = images_by_digits_by_phase["train"][d][next_image_index_by_digit[d]]
        if use_a_single_image_per_digit:
            pass  # leave the index at the first position forever
        else:
            next_image_index_by_digit[d] += 1
            if next_image_index_by_digit[d] == len(images_by_digits_by_phase["train"][d]):
                next_image_index_by_digit[d] = 0
        images_list.append(image)
    images_batch = torch.stack(images_list).to(digit_batch.device)
    return images_batch


def random_positive_or_negative_example_generator():
    if use_real_images:
        from_digit_batch_to_image_batch = get_next_real_image_batch_for_digit_batch
    else:
        from_digit_batch_to_image_batch = get_next_fake_image_batch_for_digit_batch

    def generator():
        d0_values = torch.randint(number_of_digits, (batch_size,))
        d1_values = torch.randint(number_of_digits, (batch_size,))
        i0_values = from_digit_batch_to_image_batch(d0_values)
        i1_values = from_digit_batch_to_image_batch(d1_values)
        constraint_values = (d1_values == d0_values + 1).long()
        random_pair_result = {i0: i0_values, i1: i1_values}, {constraint: constraint_values}
        # first_random_pair_result = {i0: i0_values[0], i1: i1_values[0]}, {constraint: constraint_values[0]}
        # print(first_random_pair_result)
        return random_pair_result
    return generator


def random_positive_example_generator():
    if use_real_images:
        from_digit_batch_to_image_batch = get_next_real_image_batch_for_digit_batch
    else:
        from_digit_batch_to_image_batch = get_next_fake_image_batch_for_digit_batch

    def generator():
        d0_values = torch.randint(number_of_digits - 1, (batch_size,))  # d0 is never equal to the last digit
        d1_values = d0_values + 1  # and d1 is never equal to 0
        i0_values = from_digit_batch_to_image_batch(d0_values)
        i1_values = from_digit_batch_to_image_batch(d1_values)
        random_constrained_pair_result = {i0: i0_values, i1: i1_values}, {constraint: torch.ones(batch_size).long()}
        # print(random_pair_result)
        return random_constrained_pair_result
    return generator


def after_epoch(**kwargs):
    print()
    default_learning_hook(**kwargs)
    epoch = kwargs['epoch']
    if epoch % number_of_epochs_between_evaluations == 0:
        print_digit_evaluation(**kwargs)


def print_digit_evaluation(**kwargs):
    constraint_factor, i0_d0, i1_d1 = model  # note that this relies on the factor order in the model
    from_i0_to_d0 = lambda v_i0: i0_d0.condition({i0: v_i0}).normalize().table_factor
    from_i1_to_d1 = lambda v_i1: i1_d1.condition({i1: v_i1}).normalize().table_factor
    with torch.no_grad():
        recognizers = [from_i0_to_d0] if use_shared_recognizer else [from_i0_to_d0, from_i1_to_d1]
        for recognizer in recognizers:
            print_posterior_of(recognizer, **kwargs)


def print_posterior_of(recognizer, **kwargs):
    device = kwargs.get('device')
    for digit in range(number_of_digits):
        digit_batch = torch.tensor([digit])
        image_batch = from_digit_batch_to_image_batch(digit_batch)
        if device is not None:
            digit_batch = digit_batch.to(device)
        posterior_probability = recognizer(image_batch)
        print_posterior(digit, posterior_probability)


def print_posterior(digit, output_probability):
    print(f"Prediction for \"image\" {digit}: {output_probability}")


def set_seed():
    global seed
    if seed is None:
        seed = torch.seed()
    else:
        torch.manual_seed(seed)
    print(f"Seed: {seed}")


main()
