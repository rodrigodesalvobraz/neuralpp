import pytest
import torch

from neuralpp.util.util import batch_histogram


def test_batch_histogram():
    data = [2, 5, 1, 1]
    expected = [0, 2, 1, 0, 0, 1]
    run_test(data, expected)

    data = [
        [2, 5, 1, 1],
        [3, 0, 3, 1],
    ]
    expected = [
        [0, 2, 1, 0, 0, 1],
        [1, 1, 0, 2, 0, 0],
    ]
    run_test(data, expected)

    data = [
        [[2, 5, 1, 1], [2, 4, 1, 1], ],
        [[3, 0, 3, 1], [2, 3, 1, 1], ],
    ]
    expected = [
        [[0, 2, 1, 0, 0, 1], [0, 2, 1, 0, 1, 0], ],
        [[1, 1, 0, 2, 0, 0], [0, 2, 1, 1, 0, 0], ],
    ]
    run_test(data, expected)


def test_empty_data():
    data = []
    num_classes = 2
    expected = [0, 0]
    run_test(data, expected, num_classes)

    data = [[], []]
    num_classes = 2
    expected = [[0, 0], [0, 0]]
    run_test(data, expected, num_classes)

    data = [[], []]
    run_test(data, expected=None, exception=RuntimeError)  # num_classes not provided for empty data


def run_test(data, expected, num_classes=-1, exception=None):
    data_tensor = torch.tensor(data, dtype=torch.long)

    if exception is None:
        expected_tensor = torch.tensor(expected, dtype=torch.long)
        actual = batch_histogram(data_tensor, num_classes)
        print(f"Actual: {actual}")
        assert torch.equal(actual, expected_tensor)
    else:
        with pytest.raises(exception):
            batch_histogram(data_tensor, num_classes)
