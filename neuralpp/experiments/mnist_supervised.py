import random
import time

import torch
from neuralpp.inference.graphical_model.learn.graphical_model_sgd_learner import (
    GraphicalModelSGDLearner,
)
from neuralpp.inference.graphical_model.representation.factor.neural.neural_factor import (
    NeuralFactor,
)
from neuralpp.inference.graphical_model.representation.frame.dict_frame import (
    generalized_len_of_dict_frame,
)
from neuralpp.inference.graphical_model.representation.frame.multi_frame_data_loader import (
    MultiFrameDataLoader,
)
from neuralpp.inference.graphical_model.representation.model.model import (
    compute_accuracy_on_frames_data_loader,
)
from neuralpp.inference.graphical_model.variable.integer_variable import (
    IntegerVariable,
)
from neuralpp.inference.graphical_model.variable.tensor_variable import (
    TensorVariable,
)
from neuralpp.inference.neural_net.ConvNet import ConvNet
from neuralpp.inference.neural_net.from_log_to_probabilities_adapter import (
    FromLogToProbabilitiesAdapter,
)
from neuralpp.util.generic_sgd_learner import default_after_epoch
from neuralpp.util.mnist_util import read_mnist, show_images_and_labels
from neuralpp.util.pickle_cache import pickle_cache
from neuralpp.util.util import (
    set_default_tensor_type_and_return_device,
    go_up_until_we_have_subdirectory,
)

show_examples = True
shuffle_data = True
batch_size = 100
deterministic_seed = True
debug = False
max_datapoints = 100000  # all
# max_datapoints = 5000
recompute_dataset_from_scratch = False
try_cuda = True


def main():

    if deterministic_seed:
        torch.manual_seed(0)
        random.seed(0)

    digit_var = IntegerVariable("digit_var", 10)
    image_var = TensorVariable("image_var", non_batch_dim=2)

    # Read dataset before setting default tensor type to cuda

    print("Getting NeuPP dataset ready...")
    go_up_until_we_have_subdirectory("data")
    train_dataset = pickle_cache(
        lambda: make_dataset(digit_var, image_var, "train"),
        "data/cache/simple_mnist_batcheable_train_dataset.pkl",
        refresh=recompute_dataset_from_scratch,
    )

    if show_examples:
        show_first_examples(train_dataset, 2, 3)

    device = set_default_tensor_type_and_return_device(try_cuda)
    print(f"Using {device} device")

    # Create model after setting default tensor type to cuda

    conv_net = FromLogToProbabilitiesAdapter(ConvNet())

    model = [
        NeuralFactor(
            conv_net, input_variables=[image_var], output_variable=digit_var
        ),
    ]

    print(f"Computing accuracy before training...")
    train_data_loader = MultiFrameDataLoader(train_dataset)
    accuracy = compute_accuracy_on_frames_data_loader(
        train_data_loader, model, device
    )
    print(
        f"Accuracy on training dataset before training: {accuracy*100:.2f}%"
    )

    def epoch_hook(learner):
        default_after_epoch(learner, end_str=" ")
        accuracy = compute_accuracy_on_frames_data_loader(
            learner.data_loader, model, device
        )
        time_elapsed = time.time() - learner.time_start
        print(
            f"[{time_elapsed:.0f} s] Accuracy on training dataset: {accuracy*100:.2f}%"
        )

    print("Learning...")
    GraphicalModelSGDLearner(
        model,
        train_data_loader,
        after_epoch=epoch_hook,
        debug=debug,
        device=device,
    ).learn()


def make_dataset(digit_var, image_var, phase="train"):

    print("Reading MNIST dataset...")
    images_by_digits = read_mnist(max_datapoints)

    digits = range(10)

    all_images_tensor = torch.cat(
        [torch.stack(images_by_digits[phase][d]) for d in digits]
    )
    all_digits_tensor = torch.cat(
        [
            torch.tensor([d]).repeat(len(images_by_digits[phase][d]))
            for d in digits
        ]
    )

    if shuffle_data:
        points = len(all_images_tensor)
        permutation = torch.randperm(points)
        all_permuted_images_tensor = all_images_tensor[permutation]
        all_permuted_digits_tensor = all_digits_tensor[permutation]
    else:
        all_permuted_images_tensor = all_images_tensor
        all_permuted_digits_tensor = all_digits_tensor

    dataset = [
        (
            {image_var: all_permuted_images_tensor},
            {digit_var: all_permuted_digits_tensor},
        )
    ]

    return dataset


def show_first_examples(dataset, n_rows, n_cols):
    first_observed_frame, first_query_frame = dataset[0]
    (
        images,
    ) = (
        first_observed_frame.values()
    )  # first_observed_frame is a singleton dict
    (
        labels,
    ) = first_query_frame.values()  # first_query_frame is a singleton dict
    show_images_and_labels(n_rows, n_cols, images, labels)


if __name__ == "__main__":

    main()
