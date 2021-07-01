import torch


def shape(variables):
    return tuple(v.cardinality for v in variables)


def insert_shape(original_shape, shape_to_be_inserted, dim):
    """Returns a shape resulting from inserting a given shape into an original shape at position dim"""
    shape_before_dim = original_shape[0:dim]
    shape_after_dim = original_shape[dim:]
    final_shape = shape_before_dim + shape_to_be_inserted + shape_after_dim
    return final_shape


def n_unsqueeze(tensor, n, dim):
    """Unsqueezes (that is, inserts a dimension of size 1 at the specified position dim) n times"""
    current = tensor
    for i in range(n):
        current = torch.unsqueeze(current, dim)
    return current


def index_of(elements, a_list):
    return [a_list.index(element) for element in elements]


def permutation_from_to(seq1, seq2):
    return [seq2.index(e) for e in seq1]
