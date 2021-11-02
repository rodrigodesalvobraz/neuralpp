import pytest
import torch

from neuralpp.util.util import repeat_first_dimension_with_expand, repeat_interleave_first_dimension, \
    RepeatFirstDimensionException


def test_repeat_first_dimension_with_expand():
    with pytest.raises(RepeatFirstDimensionException):
        tensor = [1,2,3]
        n = 2
        expected = None
        run_repeat_first_dimension_with_expand(tensor, n, expected)

    with pytest.raises(RepeatFirstDimensionException):
        tensor = torch.tensor(1)
        n = 2
        expected = None
        run_repeat_first_dimension_with_expand(tensor, n, expected)

    tensor = torch.tensor([1,2,3])
    n = 2
    expected = torch.tensor([1, 2, 3, 1, 2, 3])
    run_repeat_first_dimension_with_expand(tensor, n, expected)

    tensor = torch.tensor([[1,2,3], [4,5,6]])
    n = 2
    expected = torch.tensor([[1,2,3], [4,5,6], [1,2,3], [4,5,6]])
    run_repeat_first_dimension_with_expand(tensor, n, expected)

    tensor = torch.tensor([])
    n = 2
    expected = torch.zeros(0, dtype=torch.float)
    run_repeat_first_dimension_with_expand(tensor, n, expected)

    tensor = torch.tensor([])
    n = 0
    expected = torch.zeros(0, dtype=torch.float)
    run_repeat_first_dimension_with_expand(tensor, n, expected)

    tensor = torch.tensor([[1,2,3], [4,5,6]])
    n = 0
    expected = torch.zeros(0, 3, dtype=torch.long)
    run_repeat_first_dimension_with_expand(tensor, n, expected)

    tensor = torch.tensor([
        [[1,2,3],
         [4,5,6]],

        [[10,20,30],
         [40,50,60]],
    ])
    n = 2
    expected = torch.tensor([
        [[1, 2, 3],
         [4, 5, 6]],

        [[10, 20, 30],
         [40, 50, 60]],

        [[1, 2, 3],
         [4, 5, 6]],

        [[10, 20, 30],
         [40, 50, 60]],
    ])
    run_repeat_first_dimension_with_expand(tensor, n, expected)

    tensor = torch.tensor([
        [[1,2,3],
         [4,5,6]],

        [[10,20,30],
         [40,50,60]],
    ])
    n = 0
    expected = torch.zeros(0, 2, 3, dtype=torch.long)
    run_repeat_first_dimension_with_expand(tensor, n, expected)


def run_repeat_first_dimension_with_expand(tensor, n, expected):
    actual = repeat_first_dimension_with_expand(tensor, n)
    assert actual.equal(expected)


def test_repeat_interleave_first_dimension():
    with pytest.raises(RepeatFirstDimensionException):
        tensor = [1,2,3]
        n = 2
        expected = None
        run_repeat_interleave_first_dimension(tensor, n, expected)

    with pytest.raises(RepeatFirstDimensionException):
        tensor = torch.tensor(1)
        n = 2
        expected = None
        run_repeat_interleave_first_dimension(tensor, n, expected)

    tensor = torch.tensor([1,2,3])
    n = 2
    expected = torch.tensor([1,1,2,2,3,3])
    run_repeat_interleave_first_dimension(tensor, n, expected)

    tensor = torch.tensor([[1,2,3], [4,5,6]])
    n = 2
    expected = torch.tensor([[1,2,3], [1,2,3], [4,5,6], [4,5,6]])
    run_repeat_interleave_first_dimension(tensor, n, expected)

    tensor = torch.tensor([])
    n = 2
    expected = torch.zeros(0, dtype=torch.float)
    run_repeat_interleave_first_dimension(tensor, n, expected)

    tensor = torch.tensor([])
    n = 0
    expected = torch.zeros(0, dtype=torch.float)
    run_repeat_interleave_first_dimension(tensor, n, expected)

    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    n = 0
    expected = torch.zeros(0, 3, dtype=torch.long)
    run_repeat_interleave_first_dimension(tensor, n, expected)

    tensor = torch.tensor([
        [[1,2,3],
         [4,5,6]],

        [[10,20,30],
         [40,50,60]],
    ])
    n = 2
    expected = torch.tensor([
        [[1, 2, 3],
         [4, 5, 6]],

        [[1, 2, 3],
         [4, 5, 6]],

        [[10, 20, 30],
         [40, 50, 60]],

        [[10, 20, 30],
         [40, 50, 60]],
    ])
    run_repeat_interleave_first_dimension(tensor, n, expected)

    tensor = torch.tensor([
        [[1,2,3],
         [4,5,6]],

        [[10,20,30],
         [40,50,60]],
    ])
    n = 0
    expected = torch.zeros(0, 2, 3, dtype=torch.long)
    run_repeat_interleave_first_dimension(tensor, n, expected)


def run_repeat_interleave_first_dimension(tensor, n, expected):
    actual = repeat_interleave_first_dimension(tensor, n)
    assert actual.equal(expected)
