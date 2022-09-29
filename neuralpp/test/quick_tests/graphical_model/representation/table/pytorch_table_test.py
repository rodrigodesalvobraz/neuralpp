import random

import pytest
import torch
from neuralpp.inference.graphical_model.representation.table.pytorch_log_table import (
    PyTorchLogTable,
)
from neuralpp.inference.graphical_model.representation.table.pytorch_table import (
    BatchCoordinatesDoNotAgreeException,
    PyTorchTable,
)
from neuralpp.util import util
from neuralpp.util.util import check_that_exception_is_thrown, matrix_from_function


@pytest.fixture(params=["Non-log space", "Log space"])
def class_to_use(request):
    log_space = request.param == "Log space"
    if log_space:
        class_to_use = PyTorchLogTable
    else:
        class_to_use = PyTorchTable
    return class_to_use


@pytest.fixture(params=["No batch", "Empty batch", "Batch"])
def batch_size(request):
    return (
        -1
        if request.param == "No batch"
        else 0
        if request.param == "Empty batch"
        else 10
    )


@pytest.fixture(params=["No batch 1", "Empty batch 1", "Batch 1"])
def batch_size1(request):
    return (
        -1
        if request.param == "No batch 1"
        else 0
        if request.param == "Empty batch 1"
        else 10
    )


@pytest.fixture(params=["No batch 2", "Empty batch 2", "Batch 2"])
def batch_size2(request):
    return (
        -1
        if request.param == "No batch 2"
        else 0
        if request.param == "Empty batch 2"
        else 10
    )


@pytest.fixture(params=["No batch empty", "Empty batch empty", "Batch empty"])
def batch_size_empty(request):
    return (
        -1
        if request.param == "No batch empty"
        else 0
        if request.param == "Empty batch empty"
        else 10
    )


def f1_xy(x, y):
    return float(x * 2 + y)


def f1_ixy(i, x, y):
    return 10**i * f1_xy(x, y)


@pytest.fixture
def table1(class_to_use, batch_size1):

    if batch_size1 == -1:
        table1 = class_to_use.from_function((3, 2), (range(3), range(2)), f1_xy)
    else:
        table1 = class_to_use.from_function(
            (batch_size1, 3, 2),
            (range(batch_size1), range(3), range(2)),
            f1_ixy,
            batch=True,
        )
    print(f"table1: {table1}")
    return table1


@pytest.fixture
def table2(class_to_use, batch_size2):
    if batch_size2 == -1:
        table2 = class_to_use.from_array([0.7, 0.2, 0.1])
    else:
        table2 = class_to_use.from_function(
            (batch_size2, 3),
            (range(batch_size2), range(3)),
            lambda i, j: [0.7, 0.2, 0.1][j],
            batch=True,
        )
    print(f"table2: {table2}")
    return table2


@pytest.fixture
def empty(class_to_use, batch_size_empty):
    if batch_size_empty == -1:
        empty = class_to_use.from_array([])
    else:
        empty = class_to_use.from_function(
            (batch_size_empty, 0),
            (range(batch_size_empty), range(0)),
            lambda i: None,
            batch=True,
        )
    print(f"empty: {empty}")
    return empty


def test_init_table1(table1, batch_size1):
    print(f"table1: {table1}")
    if batch_size1 == -1:
        table1_data = matrix_from_function([range(3), range(2)], f1_xy)
    else:
        table1_data = matrix_from_function(
            [range(batch_size1), range(3), range(2)], f1_ixy
        )
    table1_shape = (3, 2)
    table1_length = 6
    run_init_test(batch_size1, table1, table1_length, table1_data, table1_shape)


def test_init_table2(table2, batch_size2):
    print(f"table2: {table2}")
    table2_data = [0.7, 0.2, 0.1] * (1 if batch_size2 == -1 else batch_size2)
    table2_shape = (3,)
    table2_length = 3
    run_init_test(batch_size2, table2, table2_length, table2_data, table2_shape)


def test_init_empty(empty, batch_size_empty):
    print(f"empty: {empty}")
    empty_data = []
    empty_shape = (0,)
    empty_length = 0
    run_init_test(batch_size_empty, empty, empty_length, empty_data, empty_shape)


def run_init_test(batch_size, table, table_length, table_data, table_shape):
    if batch_size == -1:
        assert torch.allclose(table.potentials_tensor(), torch.tensor(table_data))
        assert len(table) == table_length
    else:
        assert torch.allclose(
            table.potentials_tensor(),
            torch.tensor(table_data).reshape(batch_size, *(table_shape)),
        )
        assert len(table) == table_length * batch_size


def test_assignments1(table1):
    # batch does not change the set of assignments
    print(f"table1 assignments: {list(table1.assignments())}")
    assert list(table1.assignments()) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]


def test_assignments2(table2):
    # batch does not change the set of assignments
    print(f"table2 assignments: {list(table2.assignments())}")
    assert list(table2.assignments()) == [(0,), (1,), (2,)]


def test_assignments_empty(empty):
    # batch does not change the set of assignments
    print(f"empty assignments: {list(empty.assignments())}")
    assert list(empty.assignments()) == []


def test_expand1(class_to_use, table1, batch_size1):
    run_expand_test(table1, batch_size1, class_to_use)


def test_expand2(class_to_use, table2, batch_size2):
    run_expand_test(table2, batch_size2, class_to_use)


def test_expand_empty(class_to_use, empty, batch_size_empty):
    run_expand_test(empty, batch_size_empty, class_to_use)


def run_expand_test(table, batch_size, class_to_use):
    for i in range(10):
        shape_to_insert_length = random.randrange(5)
        shape_to_insert = tuple(
            random.randrange(5) for i in range(shape_to_insert_length)
        )
        dim = random.randrange(len(table.non_batch_shape))
        expected_table_expand = make_expected_expanded(
            table, shape_to_insert, dim, class_to_use, batch_size
        )
        actual_table_expand = table.expand(shape_to_insert, dim)
        assert actual_table_expand == expected_table_expand


def make_expected_expanded(table, shape_to_insert, dim, class_to_use, batch_size):
    if batch_size == -1:
        batch_dim = ()
    else:
        batch_dim = (batch_size,)
    new_shape = (
        batch_dim
        + table.non_batch_shape[:dim]
        + shape_to_insert
        + table.non_batch_shape[dim:]
    )
    new_ranges = [range(d) for d in new_shape]

    def f(*args):
        effective_dim = dim if batch_size == -1 else dim + 1
        args_without_expansion = (
            args[:effective_dim] + args[effective_dim + len(shape_to_insert) :]
        )
        return table.potentials_tensor()[args_without_expansion].item()
        # Note that we need to use table.potentials_tensor()[] rather than table[]
        # because the latter only takes non-batch coordinates into account but here we need to use the batch index.

    expected_table_expand = class_to_use.from_function(
        new_shape, new_ranges, f, batch=(batch_size != -1)
    )
    return expected_table_expand


def test_permute1(class_to_use, table1, batch_size1):
    run_permute_test(table1, batch_size1, class_to_use)


def test_permute2(class_to_use, table2, batch_size2):
    run_permute_test(table2, batch_size2, class_to_use)


def test_permute_empty(class_to_use, empty, batch_size_empty):
    run_permute_test(empty, batch_size_empty, class_to_use)


def run_permute_test(table, batch_size, class_to_use):
    for i in range(10):
        non_batch_number_of_dimensions = len(table.non_batch_shape)
        permutation = random.sample(
            range(non_batch_number_of_dimensions), non_batch_number_of_dimensions
        )
        expected_table_permutation = make_expected_permutation(
            table, permutation, class_to_use, batch_size
        )
        actual_table_permutation = table.permute(permutation)
        assert actual_table_permutation == expected_table_permutation


def make_expected_permutation(table, permutation, class_to_use, batch_size):
    if batch_size == -1:
        effective_permutation = permutation
    else:
        effective_permutation = [0] + [p + 1 for p in permutation]

    inverse_permutation = compute_inverse_permutation(effective_permutation)
    potentials_tensor = table.potentials_tensor()

    def f(*args_in_permuted_order):
        args_in_original_order = tuple(
            args_in_permuted_order[i] for i in inverse_permutation
        )
        return potentials_tensor[args_in_original_order]

    permuted_shape = permute(table.shape(), effective_permutation)
    permuted_ranges = [range(i) for i in permuted_shape]

    expected_table_permutation = class_to_use.from_function(
        permuted_shape, permuted_ranges, f, batch=(batch_size != -1)
    )
    return expected_table_permutation


def permute(array, permutation):
    return [array[i] for i in permutation]


def compute_inverse_permutation(permutation):
    result = permutation.copy()
    for i, v in enumerate(permutation):
        result[v] = i
    return result


def test_non_batch_slice1(class_to_use, table1, batch_size1):
    """
    Tests slice without batch coordinates (multiple values for the same coordinate).
    """

    test_coordinates = [
        (slice(None), slice(None)),
        (slice(None), 1),
        (1, slice(None)),
        (1, 1),
    ]

    run_non_batch_slice_tests_no_assumption_of_equal_batch_rows(
        table1,
        f1_ixy,
        table1.non_batch_shape,
        batch_size1,
        class_to_use,
        test_coordinates,
    )


def run_non_batch_slice_tests_no_assumption_of_equal_batch_rows(
    table,
    function,
    original_non_batch_shape,
    batch_size,
    class_to_use,
    test_coordinates,
):

    for coordinates in test_coordinates:
        number_of_data_rows = 1 if batch_size == -1 else batch_size
        expected_data = []

        def process_from_dim(dim, prefix_arguments):
            if dim == len(original_non_batch_shape):
                expected_data.append(function(*prefix_arguments))
            else:
                for v in range(original_non_batch_shape[dim]):
                    if coordinates[dim] == slice(None) or coordinates[dim] == v:
                        process_from_dim(dim + 1, prefix_arguments + [v])

        for row in range(number_of_data_rows):
            process_from_dim(0, prefix_arguments=[row])

        non_batch_shape = tuple(
            original_non_batch_shape[dim]
            for dim, _ in enumerate(original_non_batch_shape)
            if coordinates[dim] == slice(None)
        )

        if batch_size == -1:
            shape = tuple(non_batch_shape)
        else:
            shape = (number_of_data_rows, *non_batch_shape)
        expected_tensor = torch.tensor(expected_data).reshape(shape)
        expected = class_to_use.from_array(expected_tensor, batch=(batch_size != -1))
        assert table.slice(coordinates) == expected
        assert torch.allclose(table[coordinates], expected_tensor)


def test_non_batch_slice2(class_to_use, table2, batch_size2):
    """
    Tests slice without batch coordinates (multiple values for the same coordinate).
    """
    tests = [
        ((slice(None),), [0.7, 0.2, 0.1]),
        ((slice(0, 2),), [0.7, 0.2]),
        ((1,), 0.2),
    ]

    run_non_batch_slice_tests_assuming_equal_batch_rows(
        table2, "table2", tests, batch_size2, class_to_use
    )


def test_non_batch_slice_empty(class_to_use, empty, batch_size_empty):
    """
    Tests slice without batch coordinates (multiple values for the same coordinate).
    """
    tests = [
        ((), []),
    ]

    run_non_batch_slice_tests_assuming_equal_batch_rows(
        empty, "empty", tests, batch_size_empty, class_to_use
    )


def run_non_batch_slice_tests_assuming_equal_batch_rows(
    table, table_name, tests, batch_size, class_to_use
):
    """
    Tests slice without batch coordinates (multiple values for the same coordinate).
    """
    for coordinates, expected_row_array in tests:
        actual = table.slice(coordinates)
        print(f"{table_name}.slice(): {actual}")
        expected = make_table_from_row_array(
            class_to_use, batch_size, expected_row_array
        )
        print("expected:", expected)
        assert actual == expected


def run_non_batch_slice_tests2(table, table_name, tests, batch_size, class_to_use):
    """
    Tests slice without batch coordinates (multiple values for the same coordinate).
    """
    for coordinates, expected_non_batch_data_array in tests:
        actual = table.slice(coordinates)
        print(f"{table_name}.slice(): {actual}")
        expected = make_table_from_non_batch_data_array(
            class_to_use, batch_size, expected_non_batch_data_array
        )
        print("expected:", expected)
        assert actual == expected


def test_batch_slice1(class_to_use, table1, batch_size1):

    # Checking valid cases

    if batch_size1 == -1:
        coordinates = [1, 1]
        expected = class_to_use.from_function(
            shape=tuple(),
            function_arguments_iterables=tuple(),
            function_of_potentials=lambda: table1[1, 1],
            batch=False,
        )
        actual = table1.slice(coordinates)
        assert actual == expected
        for n in [0, 1, 10]:
            coordinates = [[1] * n, [1] * n]
            expected = class_to_use.from_function(
                shape=(n,),
                function_arguments_iterables=(range(n),),
                function_of_potentials=lambda row: f1_xy(1, 1),
                batch=True,
            )
            actual = table1.slice(coordinates)
            assert actual == expected
    elif batch_size1 == 0:
        for coordinates in [
            [1, []],
            [[], 1],
            [[], []],
        ]:
            expected = class_to_use.from_function(
                shape=(
                    0,
                ),  # only row dimension returns since non-batch dimensions would be determined.
                function_arguments_iterables=(range(0),),
                function_of_potentials=lambda row: None,
                batch=True,
            )
            actual = table1.slice(coordinates)
            assert actual == expected
    else:  # multi-row batch
        n = table1.number_of_batch_rows()
        assert n % 2 == 0, "This test assumes the batch has an even number of rows"
        for coordinates in [
            [1, 1],
            [[0, 1] * (n // 2), [0, 1] * (n // 2)],
            [1, [1] * n],
            [[1] * n, 1],
        ]:
            at_row = lambda row, c: c[row] if isinstance(c, list) else c
            expected = class_to_use.from_function(
                shape=(
                    n,
                ),  # only row dimension returns since non-batch dimensions would be determined.
                function_arguments_iterables=(range(n),),
                function_of_potentials=lambda i: f1_ixy(
                    i, at_row(i, coordinates[0]), at_row(i, coordinates[1])
                ),
                batch=True,
            )
            actual = table1.slice(coordinates)
            assert actual == expected

    # Checking invalid cases

    if batch_size1 == -1:
        coordinates = [
            [0, 1, 0],
            [0, 1],
        ]  # batch coordinates with difference numbers of values
        check_that_exception_is_thrown(
            lambda: table1.slice(coordinates), BatchCoordinatesDoNotAgreeException
        )
    elif batch_size1 == 0:
        coordinates = [1, [0, 1]]
        # batch coordinates (batch itself and second coordinate) with difference numbers of values
        check_that_exception_is_thrown(
            lambda: table1.slice(coordinates), BatchCoordinatesDoNotAgreeException
        )
    else:  # multi-row batch
        n = table1.number_of_batch_rows()
        coordinates = [
            [1] * (n // 2),
            [1] * n,
        ]  # coordinates not agreeing on number of values
        check_that_exception_is_thrown(
            lambda: table1.slice(coordinates), BatchCoordinatesDoNotAgreeException
        )
        coordinates = [
            [1] * (n // 2),
            [1] * (n // 2),
        ]  # different from neuralpp.number of batch rows
        check_that_exception_is_thrown(
            lambda: table1.slice(coordinates), BatchCoordinatesDoNotAgreeException
        )


def test_get_item2(class_to_use, table2, batch_size2):
    tests = [
        ((slice(None),), [0.7, 0.2, 0.1]),
        ((slice(0, 2),), [0.7, 0.2]),
        ((1,), 0.2),
    ]

    run_get_item_test(tests, table2, batch_size2)


def run_get_item_test(tests, table, batch_size):
    for coordinates, expected_row_array in tests:
        actual = table[coordinates]
        print(f"{coordinates}: {actual}")
        expected = make_tensor_from_row_array(batch_size, expected_row_array)
        print("expected:", expected)
        assert actual.shape == expected.shape and torch.allclose(actual, expected)


def test_mul(class_to_use):
    # It is hard to compute the expected tables without basically duplicating the multiplication code,
    # so instead here we follow the more traditional path of testing some key cases.

    # First we test some basic non-batch, non-empty cases

    table1 = class_to_use.from_array([[1.0, 2.0], [3.0, 4.0]], batch=False)
    table2 = class_to_use.from_array([[10.0, 20.0], [30.0, 40.0]], batch=False)
    expected = class_to_use.from_array([[10.0, 40.0], [90.0, 160.0]], batch=False)
    run_mul_test(expected, table1, table2)

    table1 = class_to_use.from_array([[10.0, 20.0], [30.0, 40.0]], batch=False)
    table2 = class_to_use.from_array([[1.0, 2.0], [3.0, 4.0]], batch=False)
    expected = class_to_use.from_array([[10.0, 40.0], [90.0, 160.0]], batch=False)
    run_mul_test(expected, table1, table2)

    # Now we move to some one-batch cases

    table1 = class_to_use.from_array([1.0, 2.0], batch=False)
    table2 = class_to_use.from_array([[10.0, 20.0], [30.0, 40.0]], batch=True)
    expected = class_to_use.from_array([[10.0, 40.0], [30.0, 80.0]], batch=True)
    run_mul_test(expected, table1, table2)

    table1 = class_to_use.from_array([[1.0, 2.0], [3.0, 4.0]], batch=True)
    table2 = class_to_use.from_array([10.0, 20.0], batch=False)
    expected = class_to_use.from_array([[10.0, 40.0], [30.0, 80.0]], batch=True)
    run_mul_test(expected, table1, table2)

    # We continue on to one-batch cases, but now the batches are empty
    # we use tensors to give tables the right shape.

    table1 = class_to_use.from_array([1.0, 2.0], batch=False)
    table2 = class_to_use.from_array(torch.tensor([]).reshape(0, 2), batch=True)
    expected = table2
    run_mul_test(expected, table1, table2)

    table1 = class_to_use.from_array(torch.tensor([]).reshape(0, 2), batch=True)
    table2 = class_to_use.from_array([10.0, 20.0], batch=False)
    expected = table1
    run_mul_test(expected, table1, table2)

    # Now we make even the non-batch side empty as well.

    table1 = class_to_use.from_array([], batch=False)
    table2 = class_to_use.from_array(torch.tensor([]).reshape(0, 0), batch=True)
    expected = table2
    run_mul_test(expected, table1, table2)

    table1 = class_to_use.from_array(torch.tensor([]).reshape(0, 0), batch=True)
    table2 = class_to_use.from_array([], batch=False)
    expected = table1
    run_mul_test(expected, table1, table2)

    # Now we keep the non-batch empty, but make the batches have length greater than zero
    # (still empty thought to match the non-batches).

    table1 = class_to_use.from_array([], batch=False)
    table2 = class_to_use.from_array(torch.tensor([]).reshape(10, 0), batch=True)
    expected = table2
    run_mul_test(expected, table1, table2)

    table1 = class_to_use.from_array(torch.tensor([]).reshape(10, 0), batch=True)
    table2 = class_to_use.from_array([], batch=False)
    expected = table1
    run_mul_test(expected, table1, table2)

    # And now both are empty non-batches.

    table1 = class_to_use.from_array([])
    table2 = table1
    expected = table1
    run_mul_test(expected, table1, table2)

    # Now we enter test with batches on both sizes, of varying sizes.

    # First, both batches have multiple rows

    table1 = class_to_use.from_array([[1.0, 2.0], [3.0, 5.0]], batch=True)
    table2 = class_to_use.from_array([[10.0, 20.0], [30.0, 40.0]], batch=True)
    expected = class_to_use.from_array([[10.0, 40.0], [90.0, 200.0]], batch=True)
    run_mul_test(expected, table1, table2)

    table1 = class_to_use.from_array([[10.0, 20.0], [30.0, 40.0]], batch=True)
    table2 = class_to_use.from_array([[1.0, 2.0], [3.0, 5.0]], batch=True)
    expected = class_to_use.from_array([[10.0, 40.0], [90.0, 200.0]], batch=True)
    run_mul_test(expected, table1, table2)

    # Now, both batches have zero rows

    table1 = class_to_use.from_array(torch.tensor([]).reshape(0, 2), batch=True)
    table2 = class_to_use.from_array(torch.tensor([]).reshape(0, 2), batch=True)
    expected = class_to_use.from_array(torch.tensor([]).reshape(0, 2), batch=True)
    run_mul_test(expected, table1, table2)

    # Now, both are batches but one has zero rows and the other has multiple rows
    # This should fail as the two batches must have the same number of rows
    try:
        table1 = class_to_use.from_array(torch.tensor([]).reshape(0, 2), batch=True)
        table2 = class_to_use.from_array([[1.0, 2.0], [3.0, 5.0]], batch=True)
        actual = table1 * table2
        raise AssertionError(
            "Multiplication of two batches with different number of rows should have but did not."
        )
    except AssertionError:
        pass

    # This should also fail for the same reason
    try:
        table1 = class_to_use.from_array([[1.0, 2.0]], batch=True)
        table2 = class_to_use.from_array([[1.0, 2.0], [3.0, 5.0]], batch=True)
        expected = None
        actual = table1 * table2
        raise AssertionError(
            "Multiplication of two batches with different number of rows should have but did not."
        )
    except AssertionError:
        pass


def run_mul_test(expected, table1, table2):
    actual = table1 * table2
    print("table1:", table1)
    print("table2:", table2)
    print("table1*table2:", actual)
    print("expected:", expected)
    assert actual == expected


def test_sum_out1(class_to_use, table1, batch_size1):

    if batch_size1 == -1:
        actual = table1.sum_out(0)
        expected = class_to_use.from_array([6.0, 9.0], batch=False)
        assert actual == expected

        actual = table1.sum_out(1)
        expected = class_to_use.from_array([1.0, 5.0, 9.0], batch=False)
        assert actual == expected

        actual = table1.sum_out((0, 1))
        expected = class_to_use.from_array(15.0, batch=False)
        assert actual == expected

        actual = table1.sum_out([])
        expected = class_to_use.from_array(15.0, batch=False)
        assert actual == expected

    elif batch_size1 == 0:
        actual = table1.sum_out(0)
        expected = class_to_use.from_array(torch.zeros(0, 2), batch=True)
        assert actual == expected

        actual = table1.sum_out(1)
        expected = class_to_use.from_array(torch.zeros(0, 3), batch=True)
        assert actual == expected

        actual = table1.sum_out((0, 1))
        expected = class_to_use.from_array(torch.zeros(0), batch=True)
        assert actual == expected

        actual = table1.sum_out(())
        expected = class_to_use.from_array(torch.zeros(0), batch=True)
        assert actual == expected

    elif batch_size1 > 0:
        actual = table1.sum_out(0)
        expected = class_to_use.from_array(
            [[6.0 * 10**i, 9.0 * 10**i] for i in range(batch_size1)], batch=True
        )
        assert actual == expected

        actual = table1.sum_out(1)
        expected = class_to_use.from_array(
            [[1.0 * 10**i, 5.0 * 10**i, 9.0 * 10**i] for i in range(batch_size1)],
            batch=True,
        )
        assert actual == expected

        actual = table1.sum_out((0, 1))
        expected = class_to_use.from_array(
            [15.0 * 10**i for i in range(batch_size1)], batch=True
        )
        assert actual == expected

        actual = table1.sum_out([])
        expected = class_to_use.from_array(
            [15.0 * 10**i for i in range(batch_size1)], batch=True
        )
        assert actual == expected


def test_sum_out2(class_to_use, table2, batch_size2):
    tests = [
        (0, 1.0),
        # the following does not seem consistent with the previous ones; I would expect the original tensor
        # to be preserved (no dimensions being summed out, so nothing changes), but PyTorch
        # interprets an empty list of dims to mean "all dims".
        ([], 1.0),
        ((), 1.0),
    ]

    run_sum_out_test(table2, "table2", batch_size2, class_to_use, tests)


def test_sum_out_empty(class_to_use, empty, batch_size_empty):
    tests = [
        ([], 0.0),
        ((), 0.0),
    ]

    run_sum_out_test(empty, "empty", batch_size_empty, class_to_use, tests)


def run_sum_out_test(table, table_name, batch_size, class_to_use, tests):
    for dim, expected_array in tests:
        expected = make_table_from_row_array(class_to_use, batch_size, expected_array)
        actual = table.sum_out(dim)
        print(f"{table_name}.sum_out({dim}): {actual}")
        print("expected:", expected)
        assert actual == expected


def test_sum1(class_to_use, table1, batch_size1):
    if batch_size1 == -1:
        actual = table1.sum()
        expected = torch.tensor(15.0)
        assert torch.allclose(actual, expected)

    elif batch_size1 == 0:
        actual = table1.sum()
        expected = torch.zeros(0)
        assert torch.allclose(actual, expected)

    elif batch_size1 > 0:
        actual = table1.sum()
        expected = torch.tensor([15.0 * 10**i for i in range(batch_size1)])
        assert torch.allclose(actual, expected)


def test_sum2(class_to_use, table2, batch_size2):
    run_sum_test(table2, "table2", batch_size2, expected_value=1.0)


def test_sum_empty(class_to_use, empty, batch_size_empty):
    run_sum_test(empty, "empty", batch_size_empty, expected_value=0.0)


def run_sum_test(table, table_name, batch_size1, expected_value):
    expected = make_tensor_from_row_array(batch_size1, expected_value)
    actual = table.sum()
    print(f"{table_name}:", actual)
    print("expected:", expected)
    assert actual.shape == expected.shape and torch.allclose(actual, expected)


def test_normalize1(class_to_use, table1, batch_size1):
    if batch_size1 == -1:
        actual = table1.normalize()
        print(actual)
        expected = class_to_use.from_array(
            [[0.0, 0.0667], [0.1333, 0.2], [0.2667, 0.3333]], batch=False
        )
        assert actual == expected

    elif batch_size1 == 0:
        actual = table1.normalize()
        print(actual)
        expected = class_to_use.from_array(torch.zeros(0, 3, 2), batch=True)
        assert actual == expected

    elif batch_size1 > 0:
        actual = table1.normalize()
        print(actual)
        expected = class_to_use.from_array(
            [[[0.0, 0.0667], [0.1333, 0.2], [0.2667, 0.3333]]] * batch_size1, batch=True
        )
        assert actual == expected


def test_normalize2(class_to_use, table2, batch_size2):
    # in this test we use tensors to represent expected rows as opposed to arrays like in the other test
    # because we need to divide all elements by the partition function, an operation that is not available for arrays.
    expected_row_array = (torch.tensor([0.7, 0.2, 0.1]) / 1).tolist()

    run_normalize_test(table2, "table2", batch_size2, expected_row_array, class_to_use)


def test_normalize_empty(class_to_use, empty, batch_size_empty):
    pass  # can't normalize the empty table


def run_normalize_test(table, table_name, batch_size, expected_row_array, class_to_use):
    expected = make_table_from_row_array(class_to_use, batch_size, expected_row_array)
    actual = table.normalize()
    print(f"{table_name}: {actual}")
    print("expected:", expected)
    assert actual == expected


# ###################### auxiliary functions


def close(a, b):
    return abs(a - b) < 0.0001


def make_table_from_row_array(class_to_use, batch_size, expected_row_array):
    batch = batch_size != -1
    if batch:
        # here we must create the table from neuralpp.a tensor which, unlike arrays, is able to represent a shape
        # even if the batch is empty (creating a table from neuralpp.an empty array renders its shape equal to (0)).
        non_batch_shape = util.array_shape(expected_row_array)
        total_shape = (batch_size, *non_batch_shape)
        expected_tensor = torch.tensor([expected_row_array] * batch_size).reshape(
            total_shape
        )
        expected = class_to_use.from_array(expected_tensor, batch)
    else:
        expected = class_to_use.from_array(expected_row_array, batch)
    return expected


def make_table_from_non_batch_data_array(
    class_to_use, batch_size, expected_non_batch_data_array
):
    batch = batch_size != -1
    if batch:
        non_batch_shape = util.array_shape(expected_non_batch_data_array)
        total_shape = (batch_size, *non_batch_shape)
        expected_tensor = torch.tensor(
            [expected_non_batch_data_array] * batch_size
        ).reshape(total_shape)
        expected = class_to_use.from_array(expected_tensor, batch)
    else:
        expected = class_to_use.from_array(expected_non_batch_data_array, batch)
    return expected


def make_tensor_from_row_array(batch_size, expected_row_array):
    batch = batch_size != -1
    if batch:
        non_batch_shape = util.array_shape(expected_row_array)
        total_shape = (batch_size, *non_batch_shape)
        expected = torch.tensor([expected_row_array] * batch_size).reshape(total_shape)
    else:
        expected = torch.tensor(expected_row_array)
    return expected
