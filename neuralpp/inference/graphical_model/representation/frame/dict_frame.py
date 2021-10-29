import torch
from neuralpp.util.util import generalized_len, has_len, cartesian_prod_2d


def is_frame(dictionary):
    return all(has_len(v) for v in dictionary.values())


def generalized_len_of_dict_frames(*dict_frames):
    if len(dict_frames) == 0:
        raise ThereShouldBeAtLeastOneDictFrame()
    set_of_lengths = {generalized_len_of_dict_frame(frame) for frame in dict_frames}
    if len(set_of_lengths) != 1:
        raise DictFramesShouldAllHaveTheSameLength()
    (length,) = set_of_lengths
    return length


def generalized_len_of_dict_frame(dict_frame):
    set_of_lengths = compute_set_of_lengths(dict_frame)
    if len(set_of_lengths) != 1:
        raise DictionaryValuesShouldAllHaveTheSameLength()
    (length,) = set_of_lengths
    return length


def len_of_expanded_dict_frame(dict_frame):
    """
    Same as generalized_len_of_dict_frame, but considering
    that univalues will be expanded to the length of multivalues.
    """
    set_of_lengths = compute_set_of_lengths(dict_frame)
    if len(set_of_lengths) == 1:
        length = next(iter(set_of_lengths))
    elif len(set_of_lengths) == 2 and 1 in set_of_lengths:
        length = next(l for l in set_of_lengths if l != 1)
    else:
        raise DictionaryValuesShouldAllHaveTheSameLength()
    return length


def expand_univalues_in_dict_frame(tensor_dict_frame):
    """
    Assumes dict_frame has either univalues or multivalues of the same length.
    Returns a new dict_frame with same multivalues and univalues expanded to same length as multivalues.
    """
    multivalue_lengths = compute_set_of_multivalue_lengths(tensor_dict_frame)
    if multivalue_lengths is None or len(multivalue_lengths) == 0:
        return tensor_dict_frame
    elif len(multivalue_lengths) > 1:
        raise DictFramesShouldAllHaveTheSameLength()
    else:
        multivalue_length = next(iter(multivalue_lengths))
        return \
            {
                variable: (
                    value.expand(multivalue_length)
                    if variable.multivalue_len(value) == 1
                    else value
                )
                for variable, value in tensor_dict_frame.items()
            }


def compute_set_of_lengths(dict_frame):
    if len(dict_frame) == 0:
        raise DictionaryShouldHaveAtLeastOneItem()
    # FIXME: use multivalue_len instead of generalized_len
    set_of_lengths = {generalized_len(value) for value in dict_frame.values()}
    return set_of_lengths


def compute_set_of_multivalue_lengths(dict_frame):
    if len(dict_frame) == 0:
        return None
    set_of_lengths = {variable.multivalue_len(value)
                      for variable, value in dict_frame.items()
                      if variable.is_multivalue(value)}
    return set_of_lengths


def number_of_equal_values_in_dict_frames(dict_frame1, dict_frame2):
    if dict_frame1.keys() != dict_frame2.keys():
        raise DictionariesShouldHaveTheSameKeys()
    assert_values_are_tensors(dict_frame1)
    assert_values_are_tensors(dict_frame2)
    column_comparisons = [dict_frame1[k].eq(dict_frame2[k]) for k in dict_frame1]
    comparisons_matrix = torch.stack(column_comparisons, dim=1).bool()
    number_of_equal_rows = comparisons_matrix.all(dim=1).sum().item()
    return number_of_equal_rows


def assert_values_are_tensors(dict_frame1):
    assert all(
        isinstance(v, torch.Tensor) for v in dict_frame1.values()
    ), "number_of_equal_values_in_dict_frames currently implemented for Tensor values only"


def to(dict_frame, device):
    if device is not None:
        return {v: to_if_tensor(data, device) for v, data in dict_frame.items()}
    else:
        return dict_frame


def to_if_tensor(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        return obj


def featurize_dict_frame(dict_frame):
    return {variable: variable.featurize(value) for variable, value in dict_frame.items()}


def make_cartesian_features_dict_frame(variables):
    if len(variables) > 0:
        free_cardinalities = [torch.arange(fv.cardinality) for fv in variables]
        free_assignments = cartesian_prod_2d(free_cardinalities)
        cartesian_free_features_dict_frame = {variable: free_assignments[:, i]
                                              for i, variable in enumerate(variables)}
    else:
        cartesian_free_features_dict_frame = {}
    return cartesian_free_features_dict_frame


def concatenate_into_single_tensor(dict_frame):
    if len(dict_frame) > 0:
        tensor_2d = concatenate_non_empty_dict_frame_into_single_2d_tensor(dict_frame)
    else:
        tensor_2d = torch.ones(1, 0)
    batch = any(variable.is_multivalue(value) for variable, value in dict_frame.items())
    if batch:
        tensor = tensor_2d
    else:
        tensor = tensor_2d[0]
    return tensor


def concatenate_non_empty_dict_frame_into_single_2d_tensor(dict_frame):
    """
    Given an ordered non-empty dictionary frame with all multivalue values of same length
    (but some possible univalues),
    returns a 2D tensor where element (i,j) is
    the i-th value of the j-th variable.
    """
    dict_frame_with_tensor_values = convert_frame_scalar_values_to_tensors(
        dict_frame
    )
    dict_frame_with_2d_tensor_values = convert_tensor_values_to_at_least_two_dimensions(
        dict_frame_with_tensor_values
    )
    dict_frame_with_2d_tensor_values_with_same_length = expand_tensor_values_of_len_1_to_make_all_tensors_of_same_length(
        dict_frame_with_2d_tensor_values
    )
    conditioning_tensor = torch.cat(
        tuple(dict_frame_with_2d_tensor_values_with_same_length.values()), dim=1
    )
    return conditioning_tensor


def expand_tensor_values_of_len_1_to_make_all_tensors_of_same_length(dict_frame_with_2d_tensors):
    lengths = compute_set_of_lengths(dict_frame_with_2d_tensors)
    if len(lengths) == 1:  # there is already a single length
        return dict_frame_with_2d_tensors
    elif (len(lengths) == 2 and 1 not in lengths) or len(lengths) > 2:
        raise DictionaryValuesShouldAllEitherHaveLengthOneOrSomeOtherSharedLength()
    else:
        other_length = next(iter(l for l in lengths if l != 1))
        dict_frame_with_broadcast_2d_tensors = {
            variable: value.expand(other_length, -1) if len(value) == 1 else value
            for variable, value in dict_frame_with_2d_tensors.items()
        }
        return dict_frame_with_broadcast_2d_tensors


def convert_frame_scalar_values_to_tensors(dict_frame):
    return {k: convert_to_1d_tensor_if_scalar(v) for k, v in dict_frame.items()}


def convert_to_1d_tensor_if_scalar(o):
    if isinstance(o, torch.Tensor):
        return o
    else:
        return torch.tensor([o])


def convert_tensor_values_to_at_least_two_dimensions(dict_frame):
    return {
        k: unsqueeze_if_needed_for_at_least_two_dimensions(v)
        for k, v in dict_frame.items()
    }


def unsqueeze_if_needed_for_at_least_two_dimensions(tensor):
    assert isinstance(tensor, torch.Tensor), "Must be tensor argument."
    while tensor.dim() < 2:
        tensor = tensor.unsqueeze(1)
    return tensor


def cartesian_product_of_tensor_dict_frames(dict_frame1, dict_frame2):
    n = generalized_len_of_dict_frame(dict_frame2) \
        if dict_frame2 \
        else 1
    m = generalized_len_of_dict_frame(dict_frame1) \
        if dict_frame1 \
        else 1
    repeated_expanded_featurized_dict_frame = repeat_interleave_dict_frame(dict_frame2, m)
    repeated_cartesian_features_dict_frame = repeat_dict_frame(dict_frame1, n)
    all_inputs_dict_frame = {**repeated_expanded_featurized_dict_frame, **repeated_cartesian_features_dict_frame}
    return all_inputs_dict_frame


def repeat_dict_frame(dict_frame, n):
    return {variable: value.repeat(n) for variable, value in dict_frame.items()}


def repeat_interleave_dict_frame(dict_frame, n):
    return {variable: value.repeat_interleave(n) for variable, value in dict_frame.items()}


class DictionaryShouldHaveAtLeastOneItem(BaseException):
    def __init__(self):
        super(DictionaryShouldHaveAtLeastOneItem, self).__init__(
            "Dictionary should have at least one item"
        )


class DictionaryValuesShouldAllHaveTheSameLength(BaseException):
    def __init__(self):
        super(DictionaryValuesShouldAllHaveTheSameLength, self).__init__(
            "Dictionary values should all have the same length"
        )


class DictionaryValuesShouldAllEitherHaveLengthOneOrSomeOtherSharedLength(BaseException):
    def __init__(self):
        super(DictionaryValuesShouldAllEitherHaveLengthOneOrSomeOtherSharedLength, self).__init__(
            "Dictionary values should all have either length 1 or some other shared length"
        )


class ThereShouldBeAtLeastOneDictFrame(BaseException):
    def __init__(self):
        super(ThereShouldBeAtLeastOneDictFrame, self).__init__(
            "There should be at least one dict frame"
        )


class DictFramesShouldAllHaveTheSameLength(BaseException):
    def __init__(self):
        super(DictFramesShouldAllHaveTheSameLength, self).__init__(
            "Dict frames should all have the same length"
        )


class DictionariesShouldHaveTheSameKeys(BaseException):
    def __init__(self):
        super(DictionariesShouldHaveTheSameKeys, self).__init__(
            "Dictionaries should have the same keys"
        )
