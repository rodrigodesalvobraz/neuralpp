import math
from collections import Counter

import pytest
import torch

from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import PyTorchTableFactor
from neuralpp.inference.graphical_model.representation.table.pytorch_log_table import (
    PyTorchLogTable,
)
from neuralpp.inference.graphical_model.representation.table.pytorch_table import (
    BatchCoordinatesDoNotAgreeException,
    PyTorchTable,
)
from neuralpp.inference.graphical_model.variable.discrete_variable import DiscreteVariable
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.util import util
from neuralpp.util.discrete_sampling import discrete_sample


@pytest.fixture
def x():
    return IntegerVariable("x", 3)


@pytest.fixture
def y():
    return IntegerVariable("y", 2)


@pytest.fixture
def z():
    return IntegerVariable("z", 2)


@pytest.fixture(params=["Non-log space", "Log space"])
def log_space(request):
    return request.param == "Log space"


@pytest.fixture(params=["Non-log space", "Log space"])
def log_space1(request):
    return request.param == "Log space"


@pytest.fixture(params=["Non-log space", "Log space"])
def log_space2(request):
    return request.param == "Log space"


@pytest.fixture(params=["Non-log space", "Log space"])
def log_space_expected(request):
    return request.param == "Log space"


@pytest.fixture(params=[None, 0, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[None, 0, 4])
def batch_size1(request):
    return request.param


@pytest.fixture(params=[None, 0, 4])
def batch_size2(request):
    return request.param


def test_assignments(x, y, log_space):
    factor = PyTorchTableFactor.from_function(
        (x, y), lambda x, y: float(x == y), log_space=log_space
    )
    print(list(factor.assignments()))
    assert list(factor.assignments()) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]


def test_new_instance(x, y, log_space):
    factor1 = PyTorchTableFactor.from_function(
        (x, y), lambda x, y: float(x == y), log_space=log_space
    )
    factor2 = factor1.new_instance(factor1.variables, factor1.table)
    assert factor2 is not factor1
    assert factor2 == factor2


def batch_function_adapter_from_function_without_batch_row_index(
    function_without_batch_row_index, batch
):
    def result(*args):
        args_without_batch_row_index = args[1:] if batch else args
        potential_without_batch_row_index = function_without_batch_row_index(
            *args_without_batch_row_index
        )
        if batch:
            batch_row_index = args[0]
            potential = (batch_row_index + 1) * potential_without_batch_row_index
        else:
            potential = potential_without_batch_row_index
        return potential

    return result


def batch_function_adapter_from_function_with_batch_row_index(
    function_with_batch_row_index, batch
):
    if batch:
        return function_with_batch_row_index
    else:
        return lambda *rest: function_with_batch_row_index(0, *rest)


def table_factor_from_function_without_batch_row_index(
    variables, function_without_batch_row_index, log_space, batch_size
):
    batch = batch_size is not None
    function = batch_function_adapter_from_function_without_batch_row_index(
        function_without_batch_row_index, batch
    )
    factor = PyTorchTableFactor.from_function(
        variables, function, log_space=log_space, batch_size=batch_size
    )
    return factor


def table_factor_from_function_with_batch_row_index(
    variables, function_with_batch_row_index, log_space, batch_size
):
    batch = batch_size is not None
    function = batch_function_adapter_from_function_with_batch_row_index(
        function_with_batch_row_index, batch
    )
    factor = PyTorchTableFactor.from_function(
        variables, function, log_space=log_space, batch_size=batch_size
    )
    return factor


def test_condition(x, y, log_space, log_space_expected, batch_size):
    def expected_table_factor(variables, function):
        return table_factor_from_function_with_batch_row_index(
            variables, function, log_space_expected, batch_size
        )

    fixy = lambda i, x, y: float((10 ** i) * (x * 3 + y))

    factor = table_factor_from_function_with_batch_row_index(
        (x, y), fixy, log_space, batch_size
    )

    table_class_to_use = PyTorchLogTable if log_space_expected else PyTorchTable

    tests = [
        ({}, factor),
        ({x: 0}, expected_table_factor((y,), lambda i, y: fixy(i, 0, y))),
        ({x: 1}, expected_table_factor((y,), lambda i, y: fixy(i, 1, y))),
        ({y: 0}, expected_table_factor((x,), lambda i, x: fixy(i, x, 0))),
        ({y: 1}, expected_table_factor((x,), lambda i, x: fixy(i, x, 1))),
        ({x: 0, y: 0}, expected_table_factor(tuple(), lambda i: fixy(i, 0, 0))),
        ({x: 1, y: 0}, expected_table_factor(tuple(), lambda i: fixy(i, 1, 0))),
        (
            {x: slice(None), y: 0},
            expected_table_factor((x,), lambda i, x: fixy(i, x, 0)),
        ),
        (
            {x: slice(None), y: 1},
            expected_table_factor((x,), lambda i, x: fixy(i, x, 1)),
        ),
        (
            {x: 0, y: slice(None)},
            expected_table_factor((y,), lambda i, y: fixy(i, 0, y)),
        ),
        (
            {x: 1, y: slice(None)},
            expected_table_factor((y,), lambda i, y: fixy(i, 1, y)),
        ),
        ({x: slice(None), y: slice(None)}, expected_table_factor((x, y), fixy)),
    ]

    run_condition_tests(factor, tests)

    tests_for_batch_size_different_from_zero = [
        (
            {x: [0, 2, 2, 2], y: 0},
            PyTorchTableFactor(
                tuple(),
                [fixy(0, 0, 0), fixy(0, 2, 0), fixy(0, 2, 0), fixy(0, 2, 0)],
                log_space=log_space_expected,
                batch=True,
            )
            if batch_size is None
            else PyTorchTableFactor(
                tuple(),
                [fixy(0, 0, 0), fixy(1, 2, 0), fixy(2, 2, 0), fixy(3, 2, 0)],
                log_space=log_space_expected,
                batch=True,
            )
            if batch_size == 4
            else None,
        ),
        (
            {x: [0, 2, 2, 2], y: [0, 1, 0, 1]},
            PyTorchTableFactor(
                tuple(),
                [fixy(0, 0, 0), fixy(0, 2, 1), fixy(0, 2, 0), fixy(0, 2, 1)],
                log_space=log_space_expected,
                batch=True,
            )
            if batch_size is None
            else PyTorchTableFactor(
                tuple(),
                [fixy(0, 0, 0), fixy(1, 2, 1), fixy(2, 2, 0), fixy(3, 2, 1)],
                log_space=log_space_expected,
                batch=True,
            )
            if batch_size == 4
            else None,
        ),
        (
            {x: 2, y: [0, 1, 0, 1]},
            PyTorchTableFactor(
                tuple(),
                [fixy(0, 2, 0), fixy(0, 2, 1), fixy(0, 2, 0), fixy(0, 2, 1)],
                log_space=log_space_expected,
                batch=True,
            )
            if batch_size is None
            else PyTorchTableFactor(
                tuple(),
                [fixy(0, 2, 0), fixy(1, 2, 1), fixy(2, 2, 0), fixy(3, 2, 1)],
                log_space=log_space_expected,
                batch=True,
            )
            if batch_size == 4
            else None,
        ),
    ]

    if batch_size is None or batch_size != 0:
        run_condition_tests(factor, tests_for_batch_size_different_from_zero)

    if batch_size is not None:
        illegal_conditionings_for_batches = [
            {
                x: [1] * (batch_size + 2),
                y: [1] * (batch_size + 2),
            },  # does not coincide with batch size
            # note that using batch_size + 1 would result in [1] coordinates for batch_size == 0,
            # which have length 1 and are therefore *not* considered batch_coordinates, so that would not be illegal!
            # We use batch_size + 2 to get an illegal case for both batch_size == 0 and batch_size == 4.
            {x: [0, 1], y: [0, 1, 0]},  # batch coordinates do not coincide
        ]

        for illegal_conditioning_for_batch in illegal_conditionings_for_batches:
            try:
                factor[illegal_conditioning_for_batch]
                raise AssertionError(
                    f"Should have thrown a {BatchCoordinatesDoNotAgreeException.__name__}"
                )
            except BatchCoordinatesDoNotAgreeException:
                pass

    if batch_size is None:
        illegal_conditionings_for_non_batches = [
            {x: [0, 1], y: [0, 1, 0]},  # batch coordinates do not coincide
        ]

        for illegal_conditioning_for_non_batch in illegal_conditionings_for_non_batches:
            try:
                factor[illegal_conditioning_for_non_batch]
                raise AssertionError(
                    f"Should have thrown a {BatchCoordinatesDoNotAgreeException.__name__}"
                )
            except BatchCoordinatesDoNotAgreeException:
                pass


def run_condition_tests(factor, tests):
    for assignment_dict, expected_factor in tests:
        actual_factor = factor.condition(assignment_dict)
        print(f"factor.condition({assignment_dict}) = {actual_factor}")
        assert actual_factor == expected_factor


def test_get_item(x, y, log_space, log_space_expected, batch_size):
    factor = table_factor_from_function_without_batch_row_index(
        (x, y), lambda x, y: float(x == y), log_space, batch_size
    )

    actual = factor[{x: 0, y: 0}]
    print(f"actual: {actual}")

    expected_value_0_0 = 1.0

    batch = batch_size is not None

    if batch:
        # (batch row index + 1) * expected_value_0_0
        expected = torch.tensor(
            [
                expected_value_0_0 * (batch_row_index + 1)
                for batch_row_index in range(batch_size)
            ]
        )
    else:
        expected = torch.tensor(expected_value_0_0)

    print(f"expected: {expected}")
    assert torch.allclose(actual, expected)


def test_mul(x, y, z, log_space1, log_space2, batch_size):

    f_x_y = lambda x, y: float((x + 1) * (y + 1))
    f_y_z = lambda y, z: float((y + 1) * (z + 1) * 10)

    factor1 = table_factor_from_function_without_batch_row_index(
        (x, y), f_x_y, log_space1, batch_size
    )
    factor2 = table_factor_from_function_without_batch_row_index(
        (y, z), f_y_z, log_space2, batch_size
    )

    batch = batch_size is not None

    def f_x_y_z(*args):
        if batch:
            batch_row_index, x, y, z = args
            batch_aware_f_x_y = (batch_row_index + 1) * f_x_y(x, y)
            batch_aware_f_y_z = (batch_row_index + 1) * f_y_z(y, z)
            result = batch_aware_f_x_y * batch_aware_f_y_z
            print()
            print(f"batch_row_index: {batch_row_index}")
            print(f"batch_row_index + 1: {(batch_row_index + 1)}")
            print(f"x: {x}")
            print(f"y: {y}")
            print(f"z: {z}")
            print(f"f(x,y): {f_x_y(x, y)}")
            print(f"f(y,z): {f_y_z(y, z)}")
            print(
                f"(batch_row_index + 1) * f_x_y(x,y) * (batch_row_index + 1) * f_y_z(y,z): {result}"
            )
        else:
            x, y, z = args
            result = f_x_y(x, y) * f_y_z(y, z)
            print()
            print(f"x: {x}")
            print(f"y: {y}")
            print(f"z: {z}")
            print(f"f(x,y): {f_x_y(x,y)}")
            print(f"f(y,z): {f_y_z(y,z)}")
            print(f"f_x_y(x,y) * f_y_z(y,z): {result}")

        return result

    expected_product = PyTorchTableFactor.from_function(
        (x, y, z), f_x_y_z, log_space_expected, batch_size
    )

    product = factor1 * factor2

    print(f"factor1: {factor1}")
    print(f"factor2: {factor2}")
    print(f"factor1*factor2 : {product}")
    print(f"expected product: {expected_product}")

    assert product == expected_product


def get_assignment_index(assignment, variables):
    current_stride = 1
    total = 0
    for i, v in reversed(list(enumerate(variables))):
        total += assignment[i] * current_stride
        current_stride *= v.cardinality
    return total


def test_get_assignment_index(x, y, z):
    variables = [x, y, z]
    for i, v in enumerate(variables):
        selected_variables = variables[:i]
        assignment_index = 0
        for assignment in DiscreteVariable.assignments_product(selected_variables):
            assert assignment_index == get_assignment_index(
                assignment, selected_variables
            )
            assignment_index += 1


def test_argmax(x, y, z, log_space, batch_size):
    def fixyz(i, x, y, z):
        return sum(v * 10 ** (j + 1) for j, v in enumerate([i, x, y, z]))

    factor = table_factor_from_function_with_batch_row_index(
        (x, y, z), fixyz, log_space, batch_size
    )
    if batch_size is None:
        expected = {x: 2, y: 1, z: 1}
    else:
        expected = {
            x: torch.tensor([2] * batch_size),
            y: torch.tensor([1] * batch_size),
            z: torch.tensor([1] * batch_size),
        }
    actual = factor.argmax()
    for v in (x, y, z):
        assert actual[v].eq(expected[v]).all()


def test_sample(x, y, z, log_space, batch_size):

    for number_of_variables in range(1, 3):
        variables = [x, y, z][:number_of_variables]

        number_of_assignments = math.prod(v.cardinality for v in variables)
        potentials = range(number_of_assignments)
        probabilities = torch.tensor(potentials, dtype=torch.float) / sum(potentials)

        # Create a factor in which each assignment's probability is proportional to its position
        # in the list of assignments (it's *index*).
        if batch_size is None:

            def f(*assignment):
                return get_assignment_index(assignment, variables)

        else:

            def f(*assignment):
                batch_index, assignment = assignment[0], assignment[1:]
                return get_assignment_index(assignment, variables)

        factor = PyTorchTableFactor.from_function(
            variables, f, log_space=log_space, batch_size=batch_size
        ).normalize()
        # TODO: normalized() is used for later sampling; modify Factor implementations to
        #  allow sampling without normalization

        number_of_samples = 10_000

        # We compute the maximum standard error over all values by iterating over all p_i.
        # However, for number_of_assignments >= 4 the last probability will always have the largest standard error.
        # This follows from the fact that the std error function is monotonically increasing for p <= 0.5
        # (it's an inverted parabola centered on 0.5) and, for number_of_assignments >= 4,
        # all p_i are in [0, 0.5]. Since they are always monotonically increasing,
        # the std error of the last one is the largest.
        if number_of_assignments < 4:
            max_std_err = max(std_err(p, number_of_samples) for p in probabilities)
            print(f"Searched for max std err and found {max_std_err:.3}")
        else:
            max_std_err = std_err(probabilities[-1], number_of_samples)
            print(f"Took last std err for max std err and found {max_std_err:.3}")

        z_score = (
            5  # a sample with fall out of this range extremely rarely; a Z-score of 4
            # would lead to a failure about once around every 28 years on average if we ran this once a day.
            # (A Z-score of 4 means a probability of (1 out of 10K runs) * 365 ~= 28).
        )
        absolute_tolerance = z_score * max_std_err
        print(f"Absolute tolerance is {z_score} * max error = {absolute_tolerance:.3}")

        samples = factor.sample(number_of_samples)

        if batch_size is None:
            batch_samples = samples.unsqueeze(dim=0)
        else:
            batch_samples = samples
        effective_batch_size = len(batch_samples)

        # Need to convert tensor to list of list of tensors because apparently
        # there is no way to compute histograms over batches.
        samples_per_factor_batch_row = [
            [get_assignment(batch_samples, batch_index, sample_index) for sample_index in range(number_of_samples)]
            for batch_index in range(effective_batch_size)
        ]

        for samples_for_row in samples_per_factor_batch_row:
            assignments_histogram = Counter(samples_for_row)
            assignment_indices_histogram = [assignments_histogram[a] for a in factor.assignments()]
            empirical_probabilities = util.normalize_tensor(assignment_indices_histogram)

            # Let X_i be the random indicator that the i-th assignment is sampled.
            # X_i is Bernoulli-distributed with the probability p_i that the i-th assignment is sampled.
            # The standard deviation of X_i is std_i = sqrt(p_i * (1 - p_i)).
            # The empirical probability e_i of X_i is the mean of number_of_samples samples of X_i.
            # It has, by the CLT, distribution N(p_i, std_i/sqrt(number_of_samples)).
            # std_i/sqrt(number_of_samples) is the standard error std_err_i.
            # The probability that e_i is within p_i +- 3.5 std_err_i is practically one.
            # Therefore the probability that e_i is within p_i +- max_i(std_err_i) is also practically one.

            print(f"Theoretical Probabilities: {probabilities}")
            print(f"Empirical   Probabilities: {empirical_probabilities}")
            print(f"Difference: {[probabilities[i] - empirical_probabilities[i] for i in range(len(probabilities))]}")
            print(f"Std errors: {[std_err(p, number_of_samples) for p in probabilities]}")
            print(f"Tolerance: {absolute_tolerance:.3}")

            assert torch.allclose(
                probabilities, empirical_probabilities, atol=absolute_tolerance
            ), "Samples deviated from neuralpp.expected distribution; " \
               "this is possible but should be an extremely rare event."


def get_assignment(batch_samples, batch_index, sample_index):
    return tuple(batch_samples[batch_index][sample_index].tolist())


def std_err(p, n):
    return math.sqrt(p * (1 - p) / n)


def test_torch_categorical_sampling():
    def bernoulli_std_err(p, n):
        return p * (1 - p) / math.sqrt(n)

    masses = [0, 1, 2, 3, 4, 5]
    probabilities = torch.tensor(masses, dtype=torch.float) / sum(masses)
    n = 10000
    std_errors = torch.tensor([bernoulli_std_err(p, n) for p in probabilities])
    max_std_error = max(std_errors)
    pretty_much_100_percent_range = 10 * max_std_error

    categorical = torch.distributions.Categorical(probabilities)
    pytorch_sampling_function = lambda: categorical.sample()

    def python_sampling_function():
        return discrete_sample(range(6), lambda i: probabilities[i])

    for trials in range(10):
        run_clt_test(
            python_sampling_function, probabilities, n, pretty_much_100_percent_range
        )

    for trials in range(10):
        run_clt_test(
            pytorch_sampling_function, probabilities, n, pretty_much_100_percent_range
        )


def run_clt_test(sampling_function, probabilities, n, pretty_much_100_percent_range):
    count = get_sampling_count_per_value(n, sampling_function)
    empirical_probabilities = torch.tensor(count, dtype=torch.float) / n
    print(f"     Real probabilities: {probabilities}")
    print(f"Empirical probabilities: {empirical_probabilities}")
    violation = torch.where(
        abs(empirical_probabilities - probabilities) > pretty_much_100_percent_range
    )[0]
    if len(violation) > 0:
        print(f"Violation at index {violation}")
        print(
            f"Empirical is {empirical_probabilities[violation]}, real is {probabilities[violation]}"
        )
        print(
            f"Difference is {abs(empirical_probabilities[violation] - probabilities[violation])}"
        )
        print(f"100% range is {pretty_much_100_percent_range:.10f}")
        raise Exception("CLT apparently violated")


def get_sampling_count_per_value(n, sampling_function):
    count = [0] * 6
    for i in range(n):
        sampled_value = sampling_function()
        count[sampled_value] += 1
    return count
