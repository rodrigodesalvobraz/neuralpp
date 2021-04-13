import torch


def is_batch_coordinate(coordinate):
    """
    Indicates whether given coordinate (index in a table or argument to a function) is a batch
    """
    return isinstance(coordinate, (list, range, torch.Tensor))


def contains_batch_coordinate(coordinates):
    """
    Indicates whether any of given coordinates is a batch
    """
    return any(is_batch_coordinate(c) for c in coordinates)

