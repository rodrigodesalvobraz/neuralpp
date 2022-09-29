import math

import torch


LOG_OF_NEAR_ZERO = -35.0
NEAR_ZERO = 1e-15


def log_of_nested_list_without_inf_non_differentiable(o):
    if isinstance(o, list):
        return [
            log_of_nested_list_without_inf_non_differentiable(e) for e in o
        ]
    elif isinstance(o, torch.Tensor):
        return log_without_inf_non_differentiable(o)
    else:
        try:
            if o < NEAR_ZERO:
                return LOG_OF_NEAR_ZERO
            else:
                return math.log(o)
        except Exception as e:
            print(
                f"Error in log_of_nested_list_without_inf_non_differentiable for argument {o}: {e}"
            )
            raise e


# def fix_log_of_nested_list(o):
#     if isinstance(o, list):
#         return [fix_log_of_nested_list(e) for e in o]
#     elif isinstance(o, torch.Tensor):
#         return fix_log(o)
#     elif o == -np.inf:
#         return LOG_OF_NEAR_ZERO
#     else:
#         return o


def log_without_inf_non_differentiable(tensor):
    non_zero_tensor = fix_zeros(tensor)
    result = non_zero_tensor.log()
    return result


# def fix_log(tensor):
#     tensor_clone = tensor.clone()
#     indices_of_infs = torch.isinf(tensor)
#     tensor_clone[indices_of_infs] = LOG_OF_NEAR_ZERO
#     return tensor_clone


def fix_zeros(tensor):
    if tensor.shape == ():
        return fix_zeros_in_non_dimensional_tensor(tensor)
    else:
        return fix_zeros_in_non_zero_dimensional_tensor(tensor)


def fix_zeros_in_non_dimensional_tensor(tensor):
    # I expected the code for the dimensional tensor case to work for the non-dimensional case as well,
    # but for some reason PyTorch doesn't work with that, so we do it separately here.
    if tensor == 0:
        return torch.tensor(NEAR_ZERO)
    else:
        return tensor


def fix_zeros_in_non_zero_dimensional_tensor(tensor):
    whether_it_is_zero = tensor == 0
    indices_of_zeros = whether_it_is_zero.nonzero(as_tuple=True)
    if len(indices_of_zeros) != 0:
        tensor_clone = tensor.clone()
        tensor_clone[indices_of_zeros] = NEAR_ZERO
        return tensor_clone
    else:
        return tensor


# def sum_in_log_space(x, y):
#     """
#     Given tensors x, y, computes log(e^x + e^y).
#     If x > y, then this is more accurate than the straightforward definition and also works for y = -np.inf.
#     """
#     return x + torch.log1p((y - x).log)
