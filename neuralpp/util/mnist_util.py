from itertools import islice

import torch
from neuralpp.util.pickle_cache import pickle_cache
from neuralpp.util.util import get_or_put, go_up_until_we_have_subdirectory
from torchvision import datasets, transforms
from tqdm import tqdm


read_mnist_from_original_files = False


def read_mnist(max_datapoints=None):
    go_up_until_we_have_subdirectory("data")
    return pickle_cache(
        lambda: read_mnist_no_cache(max_datapoints),
        f"data/cache/indexed_mnist_max_points={max_datapoints}.pkl",
        read_mnist_from_original_files,
    )


def read_mnist_no_cache(max_datapoints=None):
    go_up_until_we_have_subdirectory("data")
    """Return a map from phase to a map from digit to a list of its images: images_for_digit_by_phase[phase][digit]"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    phases = {"train", "test"}
    loader = {}
    images_by_digits = {}
    for p in phases:
        mnist_dataset = datasets.MNIST(
            "data", train=(p == "train"), download=True, transform=transform
        )
        loader[p] = torch.utils.data.DataLoader(mnist_dataset, batch_size=1)
        images_by_digits[p] = {}
        print(f"Indexing {p}ing MNIST dataset...")
        i = 0
        for image_singleton_batch, digit in tqdm(loader[p]):
            image = image_singleton_batch[0][
                0
            ]  # get single row of batch and single color
            images_of_digit = get_or_put(images_by_digits[p], digit.item(), [])
            images_of_digit.append(image)
            i += 1
            if max_datapoints is not None and i >= max_datapoints:
                break
    return images_by_digits


def evaluate_digits(net, get_images_of_digit, max_images_per_digit=100):
    print()
    for digit in range(10):
        number_of_correct_answers = 0
        images = get_images_of_digit(digit)[:max_images_per_digit]
        for image in images:
            prediction = net(image).argmax()
            if prediction == digit:
                number_of_correct_answers += 1
        print(f"Accuracy for {digit}: {number_of_correct_answers/len(images):.2f}")
    print()
    input("Press Enter for continuing")


def show_images_and_labels(n_rows, n_cols, images, labels):
    from matplotlib import pyplot as plt

    plt.figure()
    n_positions = n_rows * n_cols
    for i, (image, label) in enumerate(
        islice(zip(images, labels), n_positions), start=1
    ):
        plt.subplot(n_rows, n_cols, i)
        plt.tight_layout()
        plt.imshow(image, cmap="gray", interpolation="none")
        plt.title(label)
        plt.xticks([])
        plt.yticks([])
        i += 1
    plt.show()
