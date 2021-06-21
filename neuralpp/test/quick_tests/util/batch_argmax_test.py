import torch

from neuralpp.util.batch_argmax import batch_argmax, NoArgMaxIndices
from neuralpp.util.util import check_that_exception_is_thrown


def test_basic():
    # a simple array
    tensor = torch.tensor([0, 1, 2, 3, 4])
    batch_dim = 0
    expected = torch.tensor(4)
    run_test(tensor, batch_dim, expected)

    # making batch_dim = 1 renders the non-batch portion empty and argmax indices undefined
    tensor = torch.tensor([0, 1, 2, 3, 4])
    batch_dim = 1
    check_that_exception_is_thrown(lambda: batch_argmax(tensor, batch_dim), NoArgMaxIndices)

    # now a batch of arrays
    tensor = torch.tensor([[1, 2, 3], [6, 5, 4]])
    batch_dim = 1
    expected = torch.tensor([2, 0])
    run_test(tensor, batch_dim, expected)

    # Now we have an empty batch with non-batch 3-dim arrays' shape (the arrays are actually non-existent)
    tensor = torch.ones(0, 3)  # 'ones' is irrelevant since this is empty
    batch_dim = 1
    # empty batch of the right shape: just the batch dimension 0,since indices of arrays are scalar (0D)
    expected = torch.ones(0)
    run_test(tensor, batch_dim, expected)

    # Now we have an empty batch with non-batch matrices' shape (the matrices are actually non-existent)
    tensor = torch.ones(0, 3, 2)  # 'ones' is irrelevant since this is empty
    batch_dim = 1
    # empty batch of the right shape: the batch and two dimension for the indices since we have 2D matrices
    expected = torch.ones(0, 2)
    run_test(tensor, batch_dim, expected)

    # a batch of 2D matrices:
    tensor = torch.tensor([[[1, 2, 3], [6, 5, 4]], [[2, 3, 1], [4, 5, 6]]])
    batch_dim = 1
    expected = torch.tensor([[1, 0], [1, 2]])  # coordinates of two 6's, one in each 2D matrix
    run_test(tensor, batch_dim, expected)

    # same as before, but testing that batch_dim supports negative values
    tensor = torch.tensor([[[1, 2, 3], [6, 5, 4]], [[2, 3, 1], [4, 5, 6]]])
    batch_dim = -2
    expected = torch.tensor([[1, 0], [1, 2]])
    run_test(tensor, batch_dim, expected)

    # Same data, but a 2-dimensional batch of 1D arrays!
    tensor = torch.tensor([[[1, 2, 3], [6, 5, 4]], [[2, 3, 1], [4, 5, 6]]])
    batch_dim = 2
    expected = torch.tensor([[2, 0], [1, 2]])  # coordinates of 3, 6, 3, and 6
    run_test(tensor, batch_dim, expected)

    # same as before, but testing that batch_dim supports negative values
    tensor = torch.tensor([[[1, 2, 3], [6, 5, 4]], [[2, 3, 1], [4, 5, 6]]])
    batch_dim = -1
    expected = torch.tensor([[2, 0], [1, 2]])
    run_test(tensor, batch_dim, expected)


def run_test(tensor, batch_dim, expected):
    actual = batch_argmax(tensor, batch_dim)
    print(f"batch_argmax of {tensor} with batch_dim {batch_dim} is\n{actual}\nExpected:\n{expected}")
    assert actual.shape == expected.shape
    assert actual.eq(expected).all()
