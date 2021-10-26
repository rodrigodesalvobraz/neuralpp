import torch
from neuralpp.util.util import generalized_len, has_len


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


def compute_set_of_lengths(dict_frame):
    if len(dict_frame) == 0:
        raise DictionaryShouldHaveAtLeastOneItem()
    set_of_lengths = {generalized_len(values) for values in dict_frame.values()}
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


def concatenate_into_single_2d_tensor(dict_frame):
    """
    Given an ordered dictionary frame with all multivalue values of same length
    (but some possible univalues),
    returns a 2D tensor where element (i,j) is
    the j-th value of the i-th variable.
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
            variable: value.extend(other_length, -1) if len(value) == 1 else value
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
