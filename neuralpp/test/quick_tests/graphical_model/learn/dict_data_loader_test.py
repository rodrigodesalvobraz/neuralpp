from neuralpp.inference.graphical_model.representation.frame.dict_data_loader import (
    BatchSizeMustBeGreaterThanZero,
    DictDataLoader,
)
from neuralpp.inference.graphical_model.representation.frame.dict_frame import (
    DictionaryShouldHaveAtLeastOneItem,
    DictionaryValuesShouldAllHaveTheSameLength,
)
from neuralpp.inference.graphical_model.variable.integer_variable import (
    IntegerVariable,
)
from neuralpp.util.util import check_that_exception_is_thrown


def test_dict_data_loader():

    name = IntegerVariable("name", 10)
    age = IntegerVariable("age", 10)

    dictionary = {name: [1, 2, 3], age: [34, 45, 56]}
    batch_size = 2
    expected_batches = [
        {name: [1, 2], age: [34, 45]},
        {name: [3], age: [56]},
    ]
    run_dict_data_loader_test(dictionary, expected_batches, batch_size)

    dictionary = {name: [1, 2, 3], age: [34, 45, 56]}
    batch_size = 3
    expected_batches = [dictionary]
    run_dict_data_loader_test(dictionary, expected_batches, batch_size)

    dictionary = {name: [1, 2, 3], age: [34, 45, 56]}
    batch_size = 1
    expected_batches = [
        {name: [1], age: [34]},
        {name: [2], age: [45]},
        {name: [3], age: [56]},
    ]
    run_dict_data_loader_test(dictionary, expected_batches, batch_size)

    dictionary = {name: [1, 2, 3], age: [34, 45, 56]}
    batch_size = 0
    expected_batches = [
        {name: [1], age: [34]},
        {name: [2], age: [45]},
        {name: [3], age: [56]},
    ]
    check_that_exception_is_thrown(
        lambda: run_dict_data_loader_test(dictionary, expected_batches, batch_size),
        BatchSizeMustBeGreaterThanZero,
    )

    dictionary = {}
    batch_size = 3
    expected_batches = [
        {name: [1], age: [34]},
        {name: [2], age: [45]},
        {name: [3], age: [56]},
    ]
    check_that_exception_is_thrown(
        lambda: run_dict_data_loader_test(dictionary, expected_batches, batch_size),
        DictionaryShouldHaveAtLeastOneItem,
    )

    dictionary = {name: [1, 2, 3], age: [34]}
    batch_size = 3
    expected_batches = [
        {name: [1], age: [34]},
        {name: [2], age: [45]},
        {name: [3], age: [56]},
    ]
    check_that_exception_is_thrown(
        lambda: run_dict_data_loader_test(dictionary, expected_batches, batch_size),
        DictionaryValuesShouldAllHaveTheSameLength,
    )


def run_dict_data_loader_test(dictionary, expected_batches, batch_size):
    dl = DictDataLoader(dictionary, batch_size=batch_size)
    actual_batches = list(dl)
    assert actual_batches == expected_batches
