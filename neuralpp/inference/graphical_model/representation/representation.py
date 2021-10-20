import torch


def is_multivalue_coordinate(coordinate):
    """
    Indicates whether given coordinate (index in a table or argument to a function) is a batch
    """
    return isinstance(coordinate, (list, range, torch.Tensor))


def contains_multivalue_coordinate(coordinates):
    """
    Indicates whether any of given coordinates is a batch
    """
    return any(is_multivalue_coordinate(c) for c in coordinates)
