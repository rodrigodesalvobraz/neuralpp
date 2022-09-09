import math
import os
import random
from itertools import tee
from typing import List, Iterable, Any, Set, TypeVar, Dict, Iterator

import torch


T = TypeVar("T")


def split(xs, predicate):
    result = {True: [], False: []}
    for x in xs:
        result[predicate(x)].append(x)
    return result[True], result[False]


def join(xs, sep=", "):
    return sep.join([str(x) for x in xs])


def map(func, iterable):
    return (func(e) for e in iterable)


def map_of_nested_list(f, o):
    if isinstance(o, list):
        return [map_of_nested_list(f, e) for e in o]
    else:
        try:
            return f(o)
        except Exception as e:
            print(f"Error in map_of_nested_list for argument {o} and function {f}: {e}")
            raise e


def pairwise(iterable) -> Iterator:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def distinct_pairwise(iterable):
    """s -> (s0,s1), (s2,s3), (s4, s5), ..."""
    a = iter(iterable)
    return zip(a, a)


def get_or_put(dictionary, key, default):
    if key in dictionary:
        return dictionary[key]
    else:
        dictionary[key] = default
        return default


def get_or_compute_and_put(dictionary, key_base, compute_from_key_base, key_getter=lambda key_base: key_base):
    """
    Returns value stored in dict for key_getter(key_base) if present,
    or compute value (given key_base) and store it in dict under key_getter(key_base).
    key_getter default value is the identity function.
    """
    key = key_getter(key_base)
    if key in dictionary:
        return dictionary[key]
    else:
        value = compute_from_key_base(key_base)
        dictionary[key] = value
        return value


def vararg_or_array(arguments_list):
    if len(arguments_list) == 1 and type(arguments_list[0]) in {list, tuple}:
        return arguments_list[0]
    else:
        return arguments_list


def find(iterable, predicate):
    return next((x for x in iterable if predicate(x)), None)


def first(iterable):
    return next(iter(iterable), None)


def union_of_dicts(dict1, dict2):
    return {**dict1, **dict2}


def rounded_list(tensor, n_digits=3):
    return map_of_nested_list(lambda x: round(x, n_digits), tensor.tolist())


def merge_dicts(*dicts):
    return {x: d[x] for d in dicts for x in d}


def repeat(n, f):
    """Returns a list with the results of invoking f() n times"""
    return [f() for i in range(n)]


def union(iterables: Iterable[Iterable[T]]) -> Set[T]:
    result = set()
    for iterable in iterables:
        result.update(iterable)
    return result


def ordered_union_list(iterables: Iterable[Iterable[T]]) -> List[T]:
    result = []
    for iterable in iterables:
        for value in iterable:
            if value not in result:
                result.append(value)
    return result


def close(v1, v2, tolerance):
    return abs(v2 - v1) <= tolerance


def mean(sequence):
    total = 0
    length = 0
    for value in sequence:
        total += value
        length += 1
    return total / length


def select_indices_fraction(n, fraction):
    assert 0 < fraction <= 1.0, "select_indices_fraction's fraction must be in (0, 1]"
    rate = 1.0 / fraction
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
        raise Exception(
            f"check_that_exception_is_thrown received an exception instance rather than an exception type: "
            f"{exception_type}"
        )
    try:
        thunk()
        raise AssertionError(f"Should have thrown {exception_type}")
    except exception_type:
        pass
    except Exception as e:
        raise AssertionError(
            f"Should have thrown {exception_type} but instead threw {e}"
        )


def matrix_from_function(dims: List, function, prefix_index=None):
    if prefix_index is None:
        prefix_index = []
    if len(dims) == 0:
        return function(*prefix_index)
    else:
        first_dim = dims.pop(0)
        result = [
            matrix_from_function(dims, function, prefix_index=prefix_index + [index])
            for index in first_dim
        ]
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


def set_default_tensor_type_and_return_device(try_cuda, print=print):
    use_cuda = try_cuda and torch.cuda.is_available()
    if use_cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        device = torch.device("cuda")
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
        device = torch.device("cpu")
    print(f"Using {device} device")
    return device


def run_noisy_test(
        noisy_test,
        prob_spurious_failure=0.1,
        target_prob_of_unfair_rejection=0.01,
        print=print,
):
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
            raise ValueError(
                "Tolerance must be a non-negative number but was " + str(tol)
            )
        if not (expected / (1 + tol) <= actual <= expected * (1 + tol)):
            if message is None:
                message = (
                    f"{actual} not equal to {expected} within relative tolerance {tol}"
                )
            raise AssertionError(message)
    except Exception as e:
        if isinstance(e, AssertionError):
            raise e
        else:
            raise ValueError(
                f"equal_to_up_to_relative_tolerance applies to numeric values only, "
                "and tol must be a non-negative number, but got actual = {actual}, expected = {expected}, and tol = {tol}"
            )


def check_path_like(path, caller=None):
    try:
        os.fspath(path)
    except TypeError:
        raise TypeError(
            ("E" if caller is None else f"{caller} e")
            + f"xpected path-like object but got {path}"
        )


def set_seed(seed=None, print=print):
    """
    Set both Python and PyTorch seeds to the same value, using a pseudo-random one by default.
    Prints the seed at the end using given print function.
    """
    if seed is None:
        seed = torch.seed()
    else:
        torch.manual_seed(seed)
    random.seed(seed)
    print(f"Seed: {seed}")
    return seed


def go_up_until_we_have_subdirectory(subdir):
    initial_directory = os.getcwd()
    while not os.path.isdir(subdir):
        if os.getcwd() == "/":
            os.chdir(initial_directory)
            raise FileNotFoundError(f"Cannot find ancestor of '{initial_directory}' containing subdirectory '{subdir}'")
        os.chdir("..")


def value_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    else:
        return torch.tensor([value], dtype=torch.float)


def expand_into_batch(tensor, batch_size):
    """
    Returns a batch (of the given size) of replicas of given tensor.
    """
    tensor_with_newly_added_batch_dimension = tensor.unsqueeze(dim=0)
    number_of_non_batch_dimensions = tensor.dim()
    expansion_of_first_dimension = [batch_size]
    expansion_of_remaining_dimensions = [-1] * number_of_non_batch_dimensions
    expansions_by_dimensions = expansion_of_first_dimension + expansion_of_remaining_dimensions
    return tensor_with_newly_added_batch_dimension.expand(expansions_by_dimensions)


def cartesian_prod_2d(tensors):
    """
    Returns a tensor of tensors CP,
    where CP_i,j is the j-th element of the i-th tuple
    in the cartesian product of given tensors.
    This differs from torch.cartensian_prod in the case
    'tensors' is a single tensor, because in that case
    torch.cartesian_prod returns the tensor itself,
    while this function returns a tensor of tensor,
    which is consistent with the general case of multiple input tensors.
    """
    cartesian_product = torch.cartesian_prod(*tensors)
    if len(tensors) == 1:
        cartesian_product = cartesian_product.unsqueeze(
            1
        )  # to make sure cartesian_product is always 2D
    return cartesian_product


def cartesian_product_of_two_tensors(tensor1, tensor2):
    """
    Returns a tensor [ cat(tensor11,tensor21), cat(tensor11, tensor22), ..., cat(tensor1n,tensor2m)]
    where tensor1 is [tensor11, tensor12, ..., tensor1n]
    and tensor2 is [tensor21, tensor22, ..., tensor2m].
    The result is placed in the same device as tensor1.
    """
    expanded_tensor1 = tensor1.repeat_interleave(len(tensor2), dim=0)
    expanded_tensor2 = tensor2.repeat(len(tensor1), 1)
    expanded_tensor2 = expanded_tensor2.to(expanded_tensor1.device).detach()
    cartesian_product = torch.cat((expanded_tensor1, expanded_tensor2), dim=1)
    return cartesian_product


def dict_slice(dict, keys):
    return {k: v for k, v in dict.items() if k in keys}


class RepeatFirstDimensionException(BaseException):
    def __init__(self):
        super(RepeatFirstDimensionException, self).__init__(
            "repeat_first_dimension methods require tensor with at least one dimension"
        )


def _check_repeat_first_dimension_conditions(tensor):
    if not (isinstance(tensor, torch.Tensor) and tensor.dim() != 0):
        raise RepeatFirstDimensionException()


def repeat_first_dimension_with_expand(tensor, n):
    _check_repeat_first_dimension_conditions(tensor)
    original_first_dimension_length = tensor.shape[0]
    final_first_dimension_length = n * original_first_dimension_length
    final_shape = (final_first_dimension_length,) + tensor.shape[1:]
    one_d = tensor.reshape(1, tensor.numel(), )
    expanded = one_d.expand(n, *((-1,) * (one_d.dim() - 1)))
    result = expanded.reshape(final_shape)
    return result


def repeat_interleave_first_dimension(tensor, n):
    _check_repeat_first_dimension_conditions(tensor)
    return torch.repeat_interleave(tensor, n, dim=0)


def flatten_one_level(iterable, is_nested, get_nested):
    """
    Returns a list which replaces elements e in iterable where is_nested(e)
    with the elements contained in get_nested(e), while maintaining all other
    elements of iterable in order.
    """
    result = []
    for e in iterable:
        if is_nested(e):
            result.extend(get_nested(e))
        else:
            result.append(e)
    return result


def isinstance_predicate(type):
    return lambda o: isinstance(o, type)


def not_implemented(self, name):
    """
    Stub for non-implemented methods in abstract classes.
    """
    # creating a variable first prevents compiler from thinking this is an abstract method
    error = NotImplementedError(f"{name} not implemented for {type(self)}")
    raise error


def subtract(iterable, to_be_subtracted):
    return [e for e in iterable if e not in to_be_subtracted]


def all_ones_but(length, n, dim):
    if dim < 0:
        dim = length + dim
    return tuple(n if d == dim else 1 for d in range(length))


def all_minus_ones_but(length, n, dim):
    if dim < 0:
        dim = length + dim
    return tuple(n if d == dim else -1 for d in range(length))


def expand_single_dim(tensor, n, dim=0):
    return tensor.expand(all_minus_ones_but(tensor.ndim, n, dim))


def unsqueeze_and_expand(tensor, n, dim=0):
    return expand_single_dim(tensor.unsqueeze(dim), n, dim)


def fuse_k_last_dimensions_of_shape(shape, k):
    """
    Returns *shape[:-k], math.prod(shape[-k:])
    """
    return *shape[:-k], math.prod(shape[-k:])


def fuse_k_last_dimensions_of_tensor(tensor, k):
    """
    Returns tensor.reshape(fuse_k_last_dimensions_of_shape(tensor.shape, k))
    """
    return tensor.reshape(fuse_k_last_dimensions_of_shape(tensor.shape, k))


def normalize_tensor(tensor):
    normalization_constant = sum(tensor)
    empirical_probabilities = torch.tensor(tensor) / normalization_constant
    return empirical_probabilities


def batch_histogram(data_tensor, num_classes=-1):
    """
    Computes histograms of integral values, even if in batches (as opposed to torch.histc and torch.histogram).
    Arguments:
        data_tensor: a D1 x ... x D_n torch.LongTensor
        num_classes (optional): the number of classes present in data.
                                If not provided, tensor.max() + 1 is used (an error is thrown is tensor is empty).
    Returns:
        A D1 x ... x D_{n-1} x num_classes 'result' torch.LongTensor,
        containing histograms of the last dimension D_n of tensor,
        that is, result[d_1,...,d_{n-1}, c] = number of times c appears in tensor[d_1,...,d_{n-1}].
    """
    return torch.nn.functional.one_hot(data_tensor, num_classes).sum(dim=-2)


def empty(collection):
    return len(collection) == 0


def choose_elements_without_replacement(candidates_provider_function, conditions):
    """
    For each condition in conditions, selects an element from the iterable provided by
    candidates_provider_function that satisfies the condition and is not one of the previously
    selected elements.
    Note that candidates_provider_function is invoked before each selection and is allowed to return
    different iterables.
    Returns a list with selected elements (some of which may be None).
    """
    selected_elements = []
    for condition in conditions:
        candidates = [c for c in candidates_provider_function() if condition(c)]
        non_repeated_candidates = [c for c in candidates if c not in selected_elements]
        selected_element = None if empty(non_repeated_candidates) else random.choice(non_repeated_candidates)
        selected_elements.append(selected_element)
    return selected_elements


def print_dict_in_lines(dict):
    for key, value in dict.items():
        print(f"{key}: {value}")


def tensor1d_append(tensor1d, value):
    return torch.cat([tensor1d, torch.tensor([value])])


def list_for_each(
        values,
        function1=None,
        function2=None,
        filter_index=None,
        filter_element=None,
        post=None,
        post_index_result=None,
        pre=None,  # tags available if user wants to use them
        body=None):
    """
    list_for_each(values, body)
    list_for_each(values, pre, body)
    list_for_each(values, body, filter_index=None, filter_element=None, post=None)

    A function that provides an enhanced list comprehension functionality
    by giving access to the list being formed during its processing,
    as well as facilitating printing messages and other side effects.

    It receives an iterable and a function 'body', and returns
    a list of the results of applying 'body' to each element in the iterable.

    Optionally, it can receive other functions that are used at different points:
    - pre(index) is run before body if present.
    - filter_index(index) is invoked before body is applied to index.
    If present and returns a false value, this index value is skipped.
    - filter_element(element) where element is body(index),
    is invoked after body(index) but before the element is included in the resulting list.
    If present and returns a false value, the element is skipped (not included in the resulting list).
    - post(result): if present, runs after element is added to result.
    - post_index_result(index, result): if present, runs after element is added to result.
    """
    if function1 is None:
        if function2 is None:
            pass  # use pre and body named arguments if present
        else:
            raise Exception("list_for_each: first positional argument not specified but second one is")
    elif function2 is None:
        if body is None:
            body = function1
        else:  # since we have body, function1 must be pre, so try to use it:
            if pre is None:
                pre = function1
            else:
                raise Exception("list_for_each: pre function specified twice")
    else:
        if pre is None:
            pre = function1
        else:
            raise Exception("list_for_each: pre function specified twice")
        if body is None:
            body = function2
        else:
            raise Exception("list_for_each: body function specified twice")

    if body is None:
        raise Exception("list_for_each: body function not specified")

    result = []
    for index in values:
        if pre is not None:
            pre(index)
        approved = filter_index is None or filter_index(index)
        if not approved:
            continue
        element = body(index)
        approved = filter_element is None or filter_element(element)
        if approved:
            result.append(element)
        if post is not None:
            post(result)
        if post_index_result is not None:
            post_index_result(index, result)
    return result


def check_consistency_of_two_dicts(dict1: Dict, dict2: Dict):
    intersection_keys = dict1.keys() & dict2.keys()
    for key in intersection_keys:
        if dict1[key] != dict2[key]:
            raise ValueError(f"inconsistent dicts on {key}: {dict1[key]} and {dict2[key]}.")


def update_consistent_dict(dict1: Dict, dict2: Dict):
    """
    Assumes two dictionaries are consistent (i.e., for all k, if k in dict1 and k in dict2: dict1[k] == dict2[k])
    E.g., suppose dict1 = {1:2, 2:3}, after update_consistent_dict(dict1, {2:3, 3:5}), dict1 = {1:2, 2:3, 3:5};
          update_consistent_dict({1:2, 2:3}, {2:4, 3:5}) raises error.
    """
    check_consistency_of_two_dicts(dict1, dict2)
    dict1.update(dict2)


def argmax(iterable, func):
    if empty(iterable):
        return None
    return max(iterable, key=func)
