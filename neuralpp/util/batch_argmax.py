from math import prod
from typing import Tuple

import torch

from neuralpp.util.timer import Timer


def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Example:
        indices = torch.stack((torch.arange(27).long(), torch.arange(27).long())).long()
        print(unravel_indices(indices, (3, 3, 3)))
        print(unravel_indices(indices, (9, 3)))
        print(unravel_indices(indices, (9, 1, 3)))

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1).long()

    return coord


def batch_argmax(tensor, batch_dim=1):
    """
    Assumes that dimensions of tensor up to batch_dim are "batch dimensions"
    and returns the indices of the max element of each "batch row".
    More precisely, returns tensor `a` such that, for each index v of tensor.shape[:batch_dim], a[v] is
    the indices of the max element of tensor[v].
    """
    if batch_dim >= len(tensor.shape):
        raise NoArgMaxIndices()
    batch_shape = tensor.shape[:batch_dim]
    non_batch_shape = tensor.shape[batch_dim:]
    flat_non_batch_size = prod(non_batch_shape)
    tensor_with_flat_non_batch_portion = tensor.reshape(*batch_shape, flat_non_batch_size)

    dimension_of_indices = len(non_batch_shape)

    # We now have each batch row flattened in the last dimension of tensor_with_flat_non_batch_portion,
    # so we can invoke its argmax(dim=-1) method. However, that method throws an exception if the tensor
    # is empty. We cover that case first.
    if tensor_with_flat_non_batch_portion.numel() == 0:
        # If empty, either the batch dimensions or the non-batch dimensions are empty
        batch_size = prod(batch_shape)
        if batch_size == 0:  # if batch dimensions are empty
            # return empty tensor of appropriate shape
            batch_of_unraveled_indices = torch.ones(*batch_shape, dimension_of_indices).long()  # 'ones' is irrelevant as it will be empty
        else:  # non-batch dimensions are empty, so argmax indices are undefined
            raise NoArgMaxIndices()
    else:   # We actually have elements to maximize, so we search for them
        indices_of_non_batch_portion = tensor_with_flat_non_batch_portion.argmax(dim=-1)
        batch_of_unraveled_indices = unravel_indices(indices_of_non_batch_portion, non_batch_shape)

    if dimension_of_indices == 1:
        # above function makes each unraveled index of a n-D tensor a n-long tensor
        # however indices of 1D tensors are typically represented by scalars, so we squeeze them in this case.
        batch_of_unraveled_indices = batch_of_unraveled_indices.squeeze(dim=-1)
    return batch_of_unraveled_indices


class NoArgMaxIndices(BaseException):

    def __init__(self):
        super(NoArgMaxIndices, self).__init__(
            "no argmax indices: batch_argmax requires non-batch shape to be non-empty")
