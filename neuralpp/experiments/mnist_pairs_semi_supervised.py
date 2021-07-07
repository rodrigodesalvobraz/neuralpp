import torch

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
    data_loader_from_random_data_point_generator,
)
from neuralpp.util.generic_sgd_learner import default_after_epoch
from neuralpp.util.mnist_util import read_mnist, show_images_and_labels
from neuralpp.util.util import join, set_seed

# Trains a digit recognizer with the following factor graph:
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
# This script offers multiple ways of running the above, including a simplified setting
# in which the "images" are just integers from 0 to 9.
# Even in this very simple setting, the recognizer still needs to learn to associate each digit to itself.
# This provides a way of testing the general dynamic of the model without dealing with the complexities
# of actual image recognition.
#
# Other options include whether the two recognizers are the same network or not
# (essential for learning from positive examples only),
# whether negative examples are present (non-consecutive digits with Constraint = false),
# whether to use a single digit image per digit,
# and various possible initializations for the recognizer.


# -------------- PARAMETERS

number_of_digits = 10
chain_length = 10
use_real_images = True  # use real images; otherwise, use its digit value only as input (simpler version of experiment)
use_conv_net = False  # if use_real_images, neural net used is ConvNet for MNIST; otherwise, a simpler MLP for MNIST.
use_positive_examples_only = (
    True  # show only examples of consecutive pairs with the constraint labeled True,
)
# as opposed to random pairs with constraint labeled True or False accordingly.
# Including negative examples makes the problem easier, but less
# uniform random tables still get stuck in local minima.

show_examples = False  # show some examples of images (sanity check for data structure)
use_a_single_image_per_digit = (
    True  # to make the problem easier -- removes digit variability from the problem
)
try_cuda = True
batch_size = 200
number_of_batches_between_updates = (
    1  # batches have two functions: how much to fit into CUDA if using it, and
)
# how many examples to observe before updating.
# Here we are splitting those two functions, leaving batch_size for the
# number of datapoints processed at a time, but allowing for updating
# only after a number of batches are processed.
# This allows for a better estimation of gradients at each update,
# decreasing the influence of random fluctuations in these estimates for
# each update, but may make learning much slower
number_of_batches_per_epoch = 100
number_of_epochs_between_evaluations = 1
max_real_mnist_datapoints = None
number_of_digit_instances_in_evaluation = 1000
seed = None  # use None for non-deterministic seed
evaluation_probability_precision = 2

use_shared_recognizer = (
    True  # the same recognizer is shared by both "image" -> digit pairs
)
# Note that, if given positive examples only, we need a shared recognizer for
# learning to have a chance of succeeding
# because otherwise the two recognizers never get to see 9 and 0 respectively.

# 'recognizer_type' below allows the selection of different functions and initializations for recognizer.
# If use_real_images is True, this selection is ignored and a ConvNet or MLP is used, along with number_of_digits = 10.

recognizer_type = "neural net"  # if use_real_images, a ConvNet or MLP, depending on option use_conv_net
# if use_real_images == False, an MLP with a single input unit, number_of_digits hidden units,
# and number_of_digits output units.
# recognizer_type = "fixed ground truth table"  # the correct table, with fixed parameters.
# recognizer_type = "random table"  # an initialization with random potentials; see parameter below for increasing randomness
# recognizer_type = "noisy left-shift"  # a hard starting point, in which digits map to their left-shift.
# This will get most consecutive pairs to satisfy the constraint
# but for (0, 1) and (8,9), even though *every* digit is being misclassified.
# See parameter below for selecting noise level.
# Making this noisier helps because it "dillutes" the hard starting point.
recognizer_type = "uniform"  # a uniform table -- learning works well


left_shift_noise = (
    0.1  # probability of noisy left-shift recognizer initialization not left-shifting
)
# but hitting some other digit uniformly

upper_bound_for_log_potential_in_random_table = (
    1  # log of potentials are uniformly sampled from
)
# [0, upper_bound_for_log_potential_in_random_table].
# The higher the value, the farther from the uniform the table is.
# So far we observe that tables farther from the uniform
# often get stuck in local minima,
# and that uniform tables always converge to the correct answer.

# -------------- END OF PARAMETERS


lr = 1e-3 if use_real_images else 1e-3
loss_decrease_tol = lr * 0.0001

if batch_size * number_of_batches_between_updates < 500:
    max_epochs_to_go_before_stopping_due_to_loss_decrease = 15
else:
    max_epochs_to_go_before_stopping_due_to_loss_decrease = 1


class MNISTChainsProblem(LearningProblem):

    def __init__(
            self,
            number_of_digits,
            chain_length,
            use_real_images,
            use_conv_net,
            use_positive_examples_only,

            show_examples,
            use_a_single_image_per_digit,
            batch_size,
            number_of_batches_between_updates,
            number_of_batches_per_epoch,
            number_of_epochs_between_evaluations,
            max_real_mnist_datapoints,
            number_of_digit_instances_in_evaluation,
            seed,
            evaluation_probability_precision,

            use_shared_recognizer,
            recognizer_type,
            left_shift_noise,
            upper_bound_for_log_potential_in_random_table,
    ):
        self.number_of_digits = number_of_digits
        self.chain_length = chain_length
        self.use_real_images = use_real_images
        self.use_conv_net = use_conv_net
        self.use_positive_examples_only = use_positive_examples_only

        self.show_examples = show_examples
        self.use_a_single_image_per_digit = use_a_single_image_per_digit
        self.batch_size = batch_size
        self.number_of_batches_between_updates = number_of_batches_between_updates
        self.number_of_batches_per_epoch = number_of_batches_per_epoch
        self.number_of_epochs_between_evaluations = number_of_epochs_between_evaluations
        self.max_real_mnist_datapoints = max_real_mnist_datapoints
        self.number_of_digit_instances_in_evaluation = number_of_digit_instances_in_evaluation
        self.seed = seed
        self.evaluation_probability_precision = evaluation_probability_precision

        self.use_shared_recognizer = use_shared_recognizer
        self.recognizer_type = recognizer_type
        self.left_shift_noise = left_shift_noise
        self.upper_bound_for_log_potential_in_random_table = upper_bound_for_log_potential_in_random_table

        # to be defined later:
        self.images_by_digits_by_phase = None
        self.from_digit_batch_to_image_batch = None
        self.next_image_index_by_digit = None
        self.constraint_factors = None
        self.model = None
        self.images = None
        self.digits = None
        self.constraints = None

        # Adjustments:
        if self.use_real_images:
            if self.number_of_digits != 10 or self.recognizer_type != "neural net":
                print(
                    "Using real images; forcing number of digits to 10 and recognizer type to neural net"
                )
            self.number_of_digits = 10
            self.recognizer_type = "neural net"

        if self.recognizer_type == "uniform":
            self.recognizer_type = "random table"
            self.upper_bound_for_log_potential_in_random_table = (
                0  # A value of 0 samples from [0, 0], providing a uniform table.
            )

    def learning_is_needed(self):
        return self.recognizer_type != "fixed ground truth table"

    def setup_images(self):
        if self.use_real_images:
            self.setup_real_images()
        else:
            self.setup_fake_images()

    def setup_real_images(self):
        self.images_by_digits_by_phase = read_mnist(self.max_real_mnist_datapoints)
        number_of_training_images = sum(
            [
                len(self.images_by_digits_by_phase["train"][d])
                for d in range(self.number_of_digits)
            ]
        )
        print(f"Loaded {number_of_training_images:,} training images")
        self.next_image_index_by_digit = {d: 0 for d in range(self.number_of_digits)}
        self.show_digit_examples()
        self.from_digit_batch_to_image_batch = self.get_next_real_image_batch_for_digit_batch

    def setup_fake_images(self):
        self.from_digit_batch_to_image_batch = self.get_next_fake_image_batch_for_digit_batch

    def setup_model(self):
        self.create_random_variables()
        self.constraint_factors = self.make_constraint_factors()
        recognizer_factors = self.make_recognizer_factors()
        self.model = [
            # IMPORTANT: this particular factor order is relied upon later in the code
            *self.constraint_factors,
            *recognizer_factors,
        ]
        print("\nInitial model:")
        print(join(self.model, "\n"))

    def create_random_variables(self):
        self.images = []
        self.digits = []
        self.constraints = []
        for i in range(self.chain_length):
            self.images.append(
                TensorVariable(f"image{i}")
                if self.use_real_images
                else IntegerVariable(f"image{i}", self.number_of_digits)
            )
            self.digits.append(IntegerVariable(f"digit{i}", self.number_of_digits))
            if i != self.chain_length - 1:
                self.constraints.append(IntegerVariable(f"constraint{i}", 2))

    def show_digit_examples(self):
        if self.show_examples:
            images = [
                self.images_by_digits_by_phase["train"][d][i]
                for i in range(5)
                for d in range(self.number_of_digits)
            ]
            labels = [d for i in range(5) for d in range(self.number_of_digits)]
            show_images_and_labels(5, 10, images, labels)

    def make_constraint_factors(self):
        constraint_predicate = (
            lambda di, di_plus_one, constraint: int(di_plus_one == di + 1) == constraint
        )
        constraint_factors = [
            FixedPyTorchTableFactor.from_predicate(
                (self.digits[i], self.digits[i + 1], self.constraints[i]), constraint_predicate
            )
            for i in range(self.chain_length - 1)
        ]
        return constraint_factors

    def make_recognizer_factors(self):
        if self.recognizer_type == "neural net":
            if self.use_real_images:
                if self.use_conv_net:
                    print("Using ConvNet")
                    neural_net_maker = lambda: FromLogToProbabilitiesAdapter(ConvNet())
                else:
                    print("Using MLP for MNIST")

                    def neural_net_maker():
                        net = MLP(28 * 28, self.number_of_digits, self.number_of_digits)
                        return net

            else:
                neural_net_maker = lambda: MLP(1, self.number_of_digits, self.number_of_digits)

            def make_inner_function():
                neural_net = neural_net_maker()
                self.uniform_pre_training(neural_net)
                return neural_net

            def make_recognizer_factor(i, neural_net):
                return NeuralFactor(
                    neural_net, input_variables=[self.images[i]], output_variable=self.digits[i]
                )

        elif self.recognizer_type == "fixed ground truth table":
            predicate = lambda i, d: d == i

            def make_inner_function():
                return PyTorchTableFactor.from_predicate(
                    [self.images[0], self.digits[0]], predicate, log_space=True
                ).table

            def make_recognizer_factor(i, image_and_digit_table):
                return PyTorchTableFactor([self.images[i], self.digits[i]], image_and_digit_table)

        elif self.recognizer_type == "noisy left-shift":
            probability_of_left_shift = 1 - self.left_shift_noise
            number_of_non_left_shift = self.number_of_digits - 1
            probability_of_each_non_left_shift = self.left_shift_noise / number_of_non_left_shift
            left_shift_pairs = {
                (i, (i - 1) % self.number_of_digits) for i in range(self.number_of_digits)
            }

            def potential(i, d):
                return (
                    probability_of_left_shift
                    if (i, d) in left_shift_pairs
                    else probability_of_each_non_left_shift
                )

            def make_inner_function():
                return PyTorchTableFactor.from_function(
                    [self.images[0], self.digits[0]], potential, log_space=True
                ).table

            def make_recognizer_factor(i, image_and_digit_table):
                return PyTorchTableFactor([self.images[i], self.digits[i]], image_and_digit_table)

        elif self.recognizer_type == "random table":

            def make_random_parameters():
                return (
                    torch.rand(self.number_of_digits, self.number_of_digits)
                    * self.upper_bound_for_log_potential_in_random_table
                ).requires_grad_(True)

            def make_inner_function():
                return PyTorchLogTable(make_random_parameters())

            def make_recognizer_factor(i, image_and_digit_table):
                return PyTorchTableFactor([self.images[i], self.digits[i]], image_and_digit_table)

        else:
            raise Exception(f"Unknown recognizer type: {self.recognizer_type}")

        inner_functions = []
        for i in range(self.chain_length):
            inner_function = (
                inner_functions[0]
                if i != 0 and self.use_shared_recognizer
                else make_inner_function()
            )
            inner_functions.append(inner_function)

        recognizer_factors = [
            make_recognizer_factor(i, inner_function)
            for i, inner_function in enumerate(inner_functions)
        ]

        return recognizer_factors

    def make_data_loader(self):
        self.setup_images()
        batch_generator = (
            self.random_positive_examples_batch_generator()
            if self.use_positive_examples_only
            else self.random_positive_or_negative_examples_batch_generator()
        )
        train_data_loader = data_loader_from_random_data_point_generator(
            self.number_of_batches_per_epoch, batch_generator, print=None
        )
        return train_data_loader

    def get_next_fake_image_batch_for_digit_batch(self, digit_batch):
        return digit_batch

    def get_next_real_image_batch_for_digit_batch(self, digit_batch):
        images_list = []
        for d in digit_batch:
            d = d.item()
            image = self.images_by_digits_by_phase["train"][d][self.next_image_index_by_digit[d]]
            if self.use_a_single_image_per_digit:
                pass  # leave the index at the first position forever
            else:
                self.next_image_index_by_digit[d] += 1
                if self.next_image_index_by_digit[d] == len(
                    self.images_by_digits_by_phase["train"][d]
                ):
                    self.next_image_index_by_digit[d] = 0
            images_list.append(image)
        images_batch = torch.stack(images_list).to(digit_batch.device)
        return images_batch

    def random_positive_or_negative_examples_batch_generator(self):
        if self.use_real_images:
            from_digit_batch_to_image_batch = self.get_next_real_image_batch_for_digit_batch
        else:
            from_digit_batch_to_image_batch = self.get_next_fake_image_batch_for_digit_batch

        def generator():
            d_values = [
                torch.randint(self.number_of_digits, (self.batch_size,)) for i in range(self.chain_length)
            ]
            i_values = [
                from_digit_batch_to_image_batch(d_values[i]) for i in range(self.chain_length)
            ]
            constraint_values = [
                (d_values[i + 1] == d_values[i] + 1).long() for i in range(self.chain_length - 1)
            ]
            random_chain_result = (
                {self.images[i]: i_values[i] for i in range(self.chain_length)},
                {self.constraints[i]: constraint_values[i] for i in range(self.chain_length - 1)},
            )
            return random_chain_result

        return generator

    def random_positive_examples_batch_generator(self):
        if self.use_real_images:
            from_digit_batch_to_image_batch = self.get_next_real_image_batch_for_digit_batch
        else:
            from_digit_batch_to_image_batch = self.get_next_fake_image_batch_for_digit_batch

        def generator():
            d_values = []
            last_digit = self.number_of_digits - 1
            number_of_transitions_in_the_chain = self.chain_length - 1
            max_value_of_first_digit = last_digit - number_of_transitions_in_the_chain
            exclusive_upper_bound_for_first_digit = max_value_of_first_digit + 1
            d_values.append(
                torch.randint(exclusive_upper_bound_for_first_digit, (self.batch_size,))
            )
            for i in range(1, self.chain_length):
                d_values.append(d_values[i - 1] + 1)
            i_values = [
                from_digit_batch_to_image_batch(d_values[i]) for i in range(self.chain_length)
            ]
            constraint_values = [
                (d_values[i + 1] == d_values[i] + 1).long() for i in range(self.chain_length - 1)
            ]
            random_chain_result = (
                {self.images[i]: i_values[i] for i in range(self.chain_length)},
                {
                    self.constraints[i]: torch.ones(self.batch_size).long()
                    for i in range(self.chain_length - 1)
                },
            )
            return random_chain_result

        return generator

    def after_epoch(self, learner):
        print()
        default_after_epoch(learner)
        if learner.epoch % self.number_of_epochs_between_evaluations == 0:
            self.print_evaluation(learner)

    def print_evaluation(self, learner=None):
        index_of_first_recognizer_factor_in_model = (
            self.chain_length - 1
        )  # number of constraint factors
        recognizer_factor = lambda i: self.model[i + index_of_first_recognizer_factor_in_model]
        from_i_to_d = [
            lambda image: recognizer_factor(i)
            .condition({self.images[i]: image})
            .normalize()
            .table_factor
            for i in range(self.chain_length)
        ]
        with torch.no_grad():
            predictors = [from_i_to_d[0]] if self.use_shared_recognizer else from_i_to_d
            for predictor in predictors:
                self.print_posterior_of(predictor, learner)

    def print_posterior_of(self, predictor, learner=None):
        space_header = " " * len(self.make_digit_header(0))
        print(space_header + ", ".join([f"   {i}" for i in range(self.number_of_digits)]))
        for digit in range(self.number_of_digits):
            digit_batch = torch.full((self.number_of_digit_instances_in_evaluation,), digit)
            image_batch = self.from_digit_batch_to_image_batch(digit_batch)
            if learner is not None and learner.device is not None:
                image_batch = image_batch.to(learner.device)
            posterior_probability = predictor(image_batch)
            if posterior_probability.batch:
                posterior_probability_tensor = (
                    posterior_probability.table.potentials_tensor().sum(0)
                )
                posterior_probability_tensor /= posterior_probability_tensor.sum()
            else:
                posterior_probability_tensor = (
                    posterior_probability.table.potentials_tensor()
                )
            self.print_posterior_tensor(digit, posterior_probability_tensor)

    def print_posterior_tensor(self, digit, output_probability_tensor):
        digit_header = self.make_digit_header(digit)
        print(digit_header + f"{self.digit_distribution_tensor_str(output_probability_tensor)}")

    def make_digit_header(self, digit):
        image_description = "image" if self.use_real_images else 'fake "image"'
        digit_header = f"Prediction for {image_description} {digit}: "
        return digit_header

    def digit_distribution_tensor_str(self, tensor):
        return join([self.potential_str(potential) for potential in tensor])

    def potential_str(self, potential):
        if potential < 1e-2:
            return " " * (2 + self.evaluation_probability_precision)
        else:
            return f"{potential:0.{self.evaluation_probability_precision}f}"

    def uniform_pre_training(self, recognizer):
        def random_batch_generator():
            if self.use_real_images:
                from_digit_batch_to_image_batch = self.get_next_real_image_batch_for_digit_batch
            else:
                from_digit_batch_to_image_batch = self.get_next_fake_image_batch_for_digit_batch

            def generator():
                digits = torch.randint(self.number_of_digits, (self.batch_size,))
                images = from_digit_batch_to_image_batch(digits)
                return images

            return generator

        print("Uniform pre-training")
        print("Model:")
        print(recognizer)
        model = recognizer
        data_loader = data_loader_from_random_data_point_generator(
            self.number_of_batches_per_epoch, random_batch_generator(), print=None
        )
        UniformTraining(model, data_loader, self.number_of_digits).learn()
        print("Uniform pre-training completed")
        print("Model:")
        print(recognizer)


set_seed(seed)

solve_learning_problem(
    MNISTChainsProblem(
        number_of_digits,
        chain_length,
        use_real_images,
        use_conv_net,
        use_positive_examples_only,

        show_examples,
        use_a_single_image_per_digit,
        batch_size,
        number_of_batches_between_updates,
        number_of_batches_per_epoch,
        number_of_epochs_between_evaluations,
        max_real_mnist_datapoints,
        number_of_digit_instances_in_evaluation,
        seed,
        evaluation_probability_precision,

        use_shared_recognizer,
        recognizer_type,
        left_shift_noise,
        upper_bound_for_log_potential_in_random_table,
    ),
    try_cuda,
    lr,
    loss_decrease_tol,
    max_epochs_to_go_before_stopping_due_to_loss_decrease
)

