import torch

from util.util import has_len, generalized_len


def is_frame(dictionary):
    return all(has_len(v) for v in dictionary.values())


def generalized_len_of_dict_frame(dictionary):
    if len(dictionary) == 0:
        raise DictionaryShouldHaveAtLeastOneItem()
    set_of_lengths = {generalized_len(values) for values in dictionary.values()}
    if len(set_of_lengths) != 1:
        raise DictionaryValuesShouldAllHaveTheSameLength()
    length, = set_of_lengths
    return length


def generalized_len_of_dict_frames(*dict_frames):
    if len(dict_frames) == 0:
        raise ThereShouldBeAtLeastOneDictFrame()
    set_of_lengths = {generalized_len_of_dict_frame(frame) for frame in dict_frames}
    if len(set_of_lengths) != 1:
        raise DictFramesShouldAllHaveTheSameLength()
    length, = set_of_lengths
    return length


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
    assert all(isinstance(v, torch.Tensor) for v in
               dict_frame1.values()), \
        "number_of_equal_values_in_dict_frames currently implemented for Tensor values only"


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


def broadcast_values(dict_frame):
    try:
        generalized_len_of_dict_frame(dict_frame)
    except (DictionaryShouldHaveAtLeastOneItem, DictFramesShouldAllHaveTheSameLength):
        raise Exception(f"broadcast_values not implemented yet. dict_frame: {dict_frame}")
    return dict_frame


def convert_scalar_frame_to_tensor_frame(dict_frame):
    return {k: convert_scalar_to_1d_if_scalar(v) for k, v in dict_frame.items()}


def convert_scalar_to_1d_if_scalar(o):
    if isinstance(o, torch.Tensor):
        return o
    else:
        return torch.tensor([o])


def convert_values_to_at_least_two_dimensions(dict_frame):
    return {k: unsqueeze_if_needed_for_at_least_two_dimensions(v) for k, v in dict_frame.items()}


def unsqueeze_if_needed_for_at_least_two_dimensions(tensor):
    if tensor.dim() < 2:
        return tensor.unsqueeze(1)
    else:
        return tensor


class DictionaryShouldHaveAtLeastOneItem(BaseException):

    def __init__(self):
        super(DictionaryShouldHaveAtLeastOneItem, self).__init__("Dictionary should have at least one item")


class DictionaryValuesShouldAllHaveTheSameLength(BaseException):

    def __init__(self):
        super(DictionaryValuesShouldAllHaveTheSameLength, self).__init__(
            "Dictionary values should all have the same length")


class ThereShouldBeAtLeastOneDictFrame(BaseException):

    def __init__(self):
        super(ThereShouldBeAtLeastOneDictFrame, self).__init__("There should be at least one dict frame")


class DictFramesShouldAllHaveTheSameLength(BaseException):

    def __init__(self):
        super(DictFramesShouldAllHaveTheSameLength, self).__init__("Dict frames should all have the same length")


class DictionariesShouldHaveTheSameKeys(BaseException):

    def __init__(self):
        super(DictionariesShouldHaveTheSameKeys, self).__init__("Dictionaries should have the same keys")
