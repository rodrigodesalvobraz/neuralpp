from inference.graphical_model.representation.frame.dict_data_loader import DictDataLoader, BatchSizeMustBeGreaterThanZero
from inference.graphical_model.representation.frame.dict_frame import DictionaryShouldHaveAtLeastOneItem, \
    DictionaryValuesShouldAllHaveTheSameLength
from util.util import check_that_exception_is_thrown


def test_dict_data_loader():

    dictionary = {'name': ['john', 'mary', 'bob'], 'age': [34, 45, 56]}
    batch_size = 2
    expected_batches = [
        {'name': ['john', 'mary'], 'age': [34, 45]},
        {'name': ['bob'], 'age': [56]}
    ]
    run_dict_data_loader_test(dictionary, expected_batches, batch_size)

    dictionary = {'name': ['john', 'mary', 'bob'], 'age': [34, 45, 56]}
    batch_size = 3
    expected_batches = [dictionary]
    run_dict_data_loader_test(dictionary, expected_batches, batch_size)

    dictionary = {'name': ['john', 'mary', 'bob'], 'age': [34, 45, 56]}
    batch_size = 1
    expected_batches = [
        {'name': ['john'], 'age': [34]},
        {'name': ['mary'], 'age': [45]},
        {'name': ['bob'], 'age': [56]}
    ]
    run_dict_data_loader_test(dictionary, expected_batches, batch_size)

    dictionary = {'name': ['john', 'mary', 'bob'], 'age': [34, 45, 56]}
    batch_size = 0
    expected_batches = [
        {'name': ['john'], 'age': [34]},
        {'name': ['mary'], 'age': [45]},
        {'name': ['bob'], 'age': [56]}
    ]
    check_that_exception_is_thrown(lambda: run_dict_data_loader_test(dictionary, expected_batches, batch_size),
                                   BatchSizeMustBeGreaterThanZero)

    dictionary = {}
    batch_size = 3
    expected_batches = [
        {'name': ['john'], 'age': [34]},
        {'name': ['mary'], 'age': [45]},
        {'name': ['bob'], 'age': [56]}
    ]
    check_that_exception_is_thrown(lambda: run_dict_data_loader_test(dictionary, expected_batches, batch_size),
                                   DictionaryShouldHaveAtLeastOneItem)

    dictionary = {'name': ['john', 'mary', 'bob'], 'age': [34]}
    batch_size = 3
    expected_batches = [
        {'name': ['john'], 'age': [34]},
        {'name': ['mary'], 'age': [45]},
        {'name': ['bob'], 'age': [56]}
    ]
    check_that_exception_is_thrown(lambda: run_dict_data_loader_test(dictionary, expected_batches, batch_size),
                                   DictionaryValuesShouldAllHaveTheSameLength)


def run_dict_data_loader_test(dictionary, expected_batches, batch_size):
    dl = DictDataLoader(dictionary, batch_size=batch_size)
    actual_batches = list(dl)
    assert actual_batches == expected_batches
