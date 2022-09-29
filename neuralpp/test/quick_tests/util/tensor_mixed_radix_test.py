import math

import torch
from neuralpp.util.tensor_mixed_radix import (
    MaxValueException,
    TensorMixedRadix,
)


def test_tensor_mixed_radix_representation():
    values = [0, 1, 2, 3]
    radices = (2, 2)
    expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
    run_tensor_mixed_radix_representation(values, radices, expected)

    values = [0, 1, 2, 3]
    radices = (4, 4)
    expected = [(0, 0), (0, 1), (0, 2), (0, 3)]
    run_tensor_mixed_radix_representation(values, radices, expected)

    values = [0, 1, 2, 3]
    radices = (4, 2)
    expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
    run_tensor_mixed_radix_representation(values, radices, expected)

    values = [6, 7]
    radices = (4, 2)
    expected = [(3, 0), (3, 1)]
    run_tensor_mixed_radix_representation(values, radices, expected)

    values = [7, 8, 9, 10]
    radices = (2, 4, 2)
    expected = [(0, 3, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)]
    run_tensor_mixed_radix_representation(values, radices, expected)

    values = [7, 8, 16, 10]  # 16 is greater than max value 15
    radices = (2, 4, 2)
    violating_value = 16
    check_expected_exception(
        values, radices, max_value_exception(violating_value, radices)
    )

    values = [1]  # 1 is greater than max value 0
    radices = (1,)
    violating_value = 1
    check_expected_exception(
        values, radices, max_value_exception(violating_value, radices)
    )

    values = [1]  # 1 is greater than max value 0
    radices = (1, 1, 1)
    violating_value = 1
    check_expected_exception(
        values, radices, max_value_exception(violating_value, radices)
    )

    values = [1]  # 1 is greater than max value 0
    radices = ()
    violating_value = 1
    check_expected_exception(
        values, radices, max_value_exception(violating_value, radices)
    )


def max_value_exception(violating_value, radices):
    return MaxValueException(violating_value, math.prod(radices) - 1)


def check_expected_exception(values, radices, expected_exception):
    try:
        mixed_radix = TensorMixedRadix(radices)
        mixed_radix.representation(torch.tensor(values))
        raise AssertionError(f"Should have thrown {expected_exception}")
    except Exception as e:
        assert e == expected_exception


def run_tensor_mixed_radix_representation(values, radices, expected):
    mixed_radix = TensorMixedRadix(radices)
    actual = mixed_radix.representation(torch.tensor(values))
    print(f"Indices of {values} with radices {radices}: {actual}")
    assert torch.allclose(actual, torch.tensor(expected, dtype=torch.int))
