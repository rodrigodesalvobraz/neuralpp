import math
import os
from itertools import tee
from typing import List

import torch

def split(xs, predicate):
    result = {True: [], False: []}
    for x in xs:
        result[predicate(x)].append(x)
    return result[True], result[False]


def join(xs, sep=", "):
    return sep.join([str(x) for x in xs])


def map_of_nested_list(f, o):
    if isinstance(o, list):
        return [map_of_nested_list(f, e) for e in o]
    else:
        try:
            return f(o)
        except Exception as e:
            print(f"Error in map_of_nested_list for argument {o} and function {f}: {e}")
            raise e


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_or_put(dictionary, key, default):
    if key in dictionary:
        return dictionary[key]
    else:
        dictionary[key] = default
        return default


def vararg_or_array(arguments_list):
    if len(arguments_list) == 1 and type(arguments_list[0]) in {list, tuple}:
        return arguments_list[0]
    else:
        return arguments_list


def find(iterable, predicate):
    return next((x for x in iterable if predicate(x)), None)


def union_of_dicts(dict1, dict2):
    return {**dict1, **dict2}


def rounded_list(tensor, n_digits=3):
    return map_of_nested_list(lambda x: round(x, n_digits), tensor.tolist())


def merge_dicts(*dicts):
    return {x: d[x] for d in dicts for x in d}


def repeat(n, f):
    """Returns a list with the results of invoking f() n times"""
    return [f() for i in range(n)]


def union(sets):
    result = {}
    for s in sets:
        result.update(s)
    return result


def close(v1, v2, tolerance):
    return abs(v2 - v1) <= tolerance


def mean(sequence):
    total = 0
    length = 0
    for value in sequence:
        total += value
        length += 1
    return total/length


def select_indices_fraction(n, fraction):
    assert 0 < fraction <= 1.0, "select_indices_fraction's fraction must be in (0, 1]"
    rate = 1.0/fraction
    result = []
    i = 0.0
    while i < n:
        result.append(round(i))
        i += rate
    return result


def all_dims_but_first(tensor):
    num_dims = len(tensor.shape)
    return list(range(1, num_dims))


def is_iterable(object):
    try:
        iter(object)
        return True
    except Exception:
        return False


def is_empty_iterable(object):
    if is_iterable(object):
        try:
            next(iter(object))
            return False
        except StopIteration:
            return True
    else:
        return False


def map_iterable_or_value(function, iterable_of_value):
    if is_iterable(iterable_of_value):
        return [function(e) for e in iterable_of_value]
    else:
        return function(iterable_of_value)


def array_shape(array):
    """Computes the shape of tensor that would be created from given array"""
    return torch.tensor(array).shape


def tile_tensor(tensor, n, dim):
    """Replicate tensor n times along dim, without allocating memory (using torch.expand)"""

    assert 0 <= dim < len(tensor.shape), "dim must be a dimension of the given tensor"

    # this function works by unsqueezing tensor at dim (which will have size 1 and can therefore be expanded),
    # expanding dim by n, and reshaping it to the final shape, which is the original shape multiplied by n at dim.
    number_of_dimensions_after_unsqueezing = len(tensor.shape) + 1
    expand_at_dim_arg = [-1] * number_of_dimensions_after_unsqueezing
    expand_at_dim_arg[dim] = n
    final_shape = list(tensor.shape)
    final_shape[dim] *= n
    return tensor.unsqueeze(dim).expand(*expand_at_dim_arg).reshape(*final_shape)


def check_that_exception_is_thrown(thunk, exception_type):
    if isinstance(exception_type, BaseException):
        raise Exception(f"check_that_exception_is_thrown received an exception instance rather than an exception type: "
                        f"{exception_type}")
    try:
        thunk()
        raise AssertionError(f"Should have thrown {exception_type}")
    except exception_type:
        pass
    except Exception as e:
        raise AssertionError(f"Should have thrown {exception_type} but instead threw {e}")


def matrix_from_function(dims: List, function, prefix_index=None):
    if prefix_index is None:
        prefix_index = []
    if len(dims) == 0:
        return function(*prefix_index)
    else:
        first_dim = dims.pop(0)
        result = [matrix_from_function(dims, function, prefix_index=prefix_index + [index]) for index in first_dim]
        dims.insert(0, first_dim)
        return result


def has_len(obj):
    try:
        len(obj)
        return True
    except TypeError:
        return False


def generalized_len(obj):
    if has_len(obj):
        return len(obj)
    else:
        return 1


def set_default_tensor_type_and_return_device(try_cuda):
    use_cuda = try_cuda and torch.cuda.is_available()
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device("cuda")
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        device = torch.device("cpu")
    return device


def run_noisy_test(noisy_test, prob_spurious_failure=0.1, target_prob_of_unfair_rejection=0.01, print=print):
    """
    A utility for testing routines that may fail, even if correct, with a small probability (a spurious failure).
    Assumes that a success only happens if the test is indeed passing
    (therefore a single successful run is enough to conclude a passing test).
    It then proceeds to running the noisy test n times, and fails only if all n runs of noisy test fail.
    n is determined based on a wished target probability of an overall unfair rejection
    (that is, that n runs of the noisy test fail even though it is actually correct).
    """
    # Given n successive failures, probability of correct test is = prob_spurious_failure^n.
    # If we decide for overall failure,
    # the probability of being wrong (unfair rejection) is the probability of being correct.
    # prob of testing being correct = target_prob_of_unfair_rejection = p_spurious_failure^n
    # thus n = log_p target_prob_of_unfair_rejection.
    # If n is not integral, we take the ceiling.
    n = math.ceil(math.log(target_prob_of_unfair_rejection, prob_spurious_failure))
    try_noisy_test_up_to_n_times(noisy_test, n, print)


def try_noisy_test_up_to_n_times(noisy_test, n=3, print=print):
    """
    A utility for testing routines that may fail, even if correct, with a small probability (a spurious failure).
    It attempts to get a successful run up to 3 times, and fails in case that does not happen.
    """
    failure_value = None
    for i in range(n):
        try:
            failure_value = noisy_test()
            if failure_value is True:
                failure_value = None
        except AssertionError as e:
            failure_value = e

        if failure_value is None:
            return True

        if print is not None:
            print("Test failed: " + str(failure_value))
            if i < n - 1:
                print("That may have been spurious, so will try again.")

    if isinstance(failure_value, Exception):
        raise failure_value
    else:
        return failure_value


def assert_equal_up_to_relative_tolerance(actual, expected, tol, message=None):
    """
    Compares two numeric values up to a relative tolerance (a non-negative number).
    Succeeds if expected/(1 + tol) <= actual <= expected*(1 + tol)
    """
    try:
        if tol < 0:
            raise ValueError("Tolerance must be a non-negative number but was " + str(tol))
        if not (expected/(1 + tol) <= actual <= expected*(1 + tol)):
            if message is None:
                message = f"{actual} not equal to {expected} within relative tolerance {tol}"
            raise AssertionError(message)
    except Exception as e:
        if isinstance(e, AssertionError):
            raise e
        else:
            raise ValueError(f"equal_to_up_to_relative_tolerance applies to numeric values only, "
                             "and tol must be a non-negative number, but got actual = {actual}, expected = {expected}, and tol = {tol}")


def check_path_like(path, caller=None):
    try:
        os.fspath(path)
    except TypeError:
        raise TypeError(("E" if caller is None else f"{caller} e") + f"xpected path-like object but got {path}")
