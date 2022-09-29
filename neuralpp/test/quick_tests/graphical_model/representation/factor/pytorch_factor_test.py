import random

import pytest
import torch

from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable


@pytest.fixture(params=["Non-log space", "Log space"])
def log_space(request):
    return request.param == "Log space"


@pytest.fixture(params=["No batch", "Empty batch", "Batch"])
def batch_size(request):
    return (
        -1
        if request.param == "No batch"
        else 0
        if request.param == "Empty batch"
        else 10
    )


@pytest.fixture
def x():
    return IntegerVariable("x", 3)


@pytest.fixture
def y():
    return IntegerVariable("y", 2)


@pytest.fixture
def factor1(x, y, log_space):
    # noinspection PyShadowingNames
    result = PyTorchTableFactor.from_function(
        [x, y], lambda x, y: 0.9 if x == y else 0.1, log_space=log_space
    )
    print(f"factor1: {result}")
    return result


@pytest.fixture
def factor2(x, y, log_space):
    # noinspection PyShadowingNames
    result = PyTorchTableFactor([x], [0.7, 0.2, 0.1], log_space=log_space)
    print(f"factor1: {result}")
    return result


@pytest.fixture
def normalized_product_factor(factor1, factor2):
    print("Product:", factor1 * factor2)
    normalized_product = (factor1 * factor2).normalize()
    print("Normalized product:", normalized_product)
    return normalized_product


def test_sample(normalized_product_factor):

    random.seed(2)
    torch.manual_seed(2)

    print("Samples from neuralpp.normalized product:")
    actual = [
        normalized_product_factor.sample() for i in range(20)
    ]  # TODO: use batch sampling when available
    expected = [
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
    ]
    print(f"Expected: {expected}")
    print(f"Actual  : {actual}")
    assert all((torch.equal(a, e) for a, e in zip(actual, expected)))


def test_normalization(x, y, factor1, factor2):
    product = factor1 * factor2
    print("Product:", product)
    normalized_product = product.normalize()
    print("Normalized product:", normalized_product)
    expected = PyTorchTableFactor(
        [y, x], [[0.6848, 0.0217, 0.0109], [0.0761, 0.1957, 0.0109]]
    )
    assert normalized_product == expected


def test_condition(x, y, factor1, factor2):
    conditioned_nothing = factor1.condition({})
    print(f"factor1 | nothing: {conditioned_nothing}")
    expected = factor1
    assert conditioned_nothing == expected

    conditioned_x_0 = factor1.condition({x: 0})
    print(f"factor1 | x = 0: {conditioned_x_0}")
    expected = PyTorchTableFactor([y], [0.9, 0.1])
    assert conditioned_x_0 == expected

    conditioned_y_0 = factor1.condition({y: 0})
    print(f"factor1 | y = 0: {conditioned_y_0}")
    expected = PyTorchTableFactor([x], [0.9, 0.1, 0.1])
    assert conditioned_y_0 == expected

    conditioned_x_0_y_1 = factor1.condition({x: 0, y: 1})
    print(f"factor1 | x = 0, y = 1: {conditioned_x_0_y_1}")
    expected = PyTorchTableFactor([], 0.1)
    assert conditioned_x_0_y_1 == expected


def test_sum_out(x, y, factor1, factor2):
    print(f"factor1 ^ x: {factor1 ^ x}")
    print(f"factor1 ^ y: {factor1 ^ y}")
    print(f"factor1 ^ [x, y]: {factor1 ^ [x, y]}")
    print(f"factor2 ^ x: {factor2 ^ x}")
    assert PyTorchTableFactor([y], [1.1, 1.1]) == factor1 ^ x
    assert PyTorchTableFactor([x], [1.0, 1.0, 0.2]) == factor1 ^ y
    assert PyTorchTableFactor([], 2.2) == factor1 ^ [x, y]
    assert PyTorchTableFactor([], 1.0) == factor2 ^ x


def test_zeros():
    x = IntegerVariable("x", 2)
    y = IntegerVariable("y", 2)
    for log_space in {True, False}:
        f = PyTorchTableFactor.from_function(
            [x, y], lambda x, y: float(x == y), log_space=log_space
        )
        assert close(f({x: 0, y: 0}), 1.0)
        assert close(f({x: 0, y: 1}), 0.0)
        assert close(f({x: 1, y: 0}), 0.0)
        assert close(f({x: 1, y: 1}), 1.0)


def close(a, b):
    return abs(a - b) < 0.0001
