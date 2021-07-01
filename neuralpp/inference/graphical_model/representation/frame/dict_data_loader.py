from typing import Any, Dict

from neuralpp.inference.graphical_model.representation.frame.dict_frame import (
    generalized_len_of_dict_frame,
)


def dict_data_generator(dictionary: Dict[Any, Any], length, batch_size=100):
    for i in range(0, length, batch_size):
        yield {k: v[i : i + batch_size] for k, v in dictionary.items()}


class DictDataLoader:
    """
    A data loader for datasets organized as a dict, where each key maps to an array-like set of values.
    The data loader returns dicts with the same keys and batches of the array-like values.
    """

    def __init__(self, dictionary: Dict[Any, Any], batch_size=100):
        self.dictionary = dictionary
        self.batch_size = batch_size
        self.length = generalized_len_of_dict_frame(dictionary)
        if batch_size <= 0:
            raise BatchSizeMustBeGreaterThanZero()

    def __iter__(self):
        return dict_data_generator(self.dictionary, self.length, self.batch_size)

    def __len__(self):
        return self.length


class BatchSizeMustBeGreaterThanZero(BaseException):
    def __init__(self):
        super(BatchSizeMustBeGreaterThanZero, self).__init__(
            "Batch size must be greater than zero"
        )
