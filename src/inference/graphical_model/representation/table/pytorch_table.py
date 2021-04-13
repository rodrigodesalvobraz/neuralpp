import itertools
import math

import torch
from torch.distributions import Categorical

from inference.graphical_model.representation.representation import is_batch_coordinate, contains_batch_coordinate
from inference.graphical_model.representation.table.table import Table
from inference.graphical_model.representation.table.table_util import insert_shape, n_unsqueeze
from util import util
from util.batch_argmax import batch_argmax
from util.first import first
from util.tensor_mixed_radix import TensorMixedRadix
from util.util import map_of_nested_list, all_dims_but_first, is_iterable


class PyTorchTable(Table):

    def __init__(self, raw_entries, batch=False):
        super().__init__()
        self.raw_tensor = \
            raw_entries \
            if isinstance(raw_entries, torch.Tensor) \
            else torch.tensor(raw_entries, requires_grad=True)
        self.batch = batch
        if self.batch:
            self.non_batch_shape = self.raw_tensor.shape[1:]
        else:
            self.non_batch_shape = self.raw_tensor.shape

        self._non_batch_radices = None

    def new_table_from_raw_entries(self, raw_entries, batch=False):
        return type(self)(raw_entries, batch)

    def shape(self):
        return self.raw_tensor.shape

    def number_of_batch_rows(self):
        """Returns the number of batch rows or None if table is not a batch"""
        if self.batch:
            return self.raw_tensor.shape[0]
        else:
            return None

    @staticmethod
    def from_array(array_of_potentials, batch=False):
        return PyTorchTable(array_of_potentials, batch)

    @staticmethod
    def from_function(shape, function_arguments_iterables, function_of_potentials, batch=False):

        array_of_potentials = []
        for function_arguments_tuple in itertools.product(*function_arguments_iterables):
            potential = function_of_potentials(*function_arguments_tuple)
            array_of_potentials.append(float(potential))

        # array_of_potentials = [float(function_of_potentials(*function_arguments_tuple))
        #                        for function_arguments_tuple in itertools.product(*function_arguments_iterables)]

        one_dim_tensor = torch.tensor(array_of_potentials)
        right_shape_tensor_of_potentials = one_dim_tensor.reshape(shape)
        return PyTorchTable(right_shape_tensor_of_potentials, batch)

    # Methods depending on structure only (not values)
    # This distinction is useful if a sub-class represents values in some other system
    # (such as in log-space), because then we know these methods need not be overridden.

    def __len__(self):
        return self.raw_tensor.numel()

    def assignments(self):
        return itertools.product(*[range(dim) for dim in self.non_batch_shape])

    def expand(self, shape_to_be_inserted, dim):
        """
        Inserts new dimensions with a given shape at specific position dim
        (after disregarding possible batch dimension)
        """
        effective_dim = dim + 1 if self.batch else dim
        unsqueezed_tensor = n_unsqueeze(self.raw_tensor, len(shape_to_be_inserted), effective_dim)
        final_shape = insert_shape(self.raw_tensor.shape, shape_to_be_inserted, effective_dim)
        new_tensor = unsqueezed_tensor.expand(final_shape)
        return self.new_table_from_raw_entries(new_tensor, self.batch)

    def permute(self, permutation):
        """
        Applies permutation (of non-batch dimensions).
        """
        effective_permutation = [0] + [p + 1 for p in permutation] if self.batch else permutation
        permuted_raw_tensor = self.raw_tensor.permute(effective_permutation)
        return self.new_table_from_raw_entries(permuted_raw_tensor, self.batch)

    def pytorch_parameters(self):
        return [self.raw_tensor]

    def get_raw_tensor_slice(self, non_batch_slice_coordinates):
        # see the documentation for Table.__getitem__ to better understand this implementation
        if is_iterable(non_batch_slice_coordinates):
            tuple_of_non_batch_slice_coordinates = tuple(non_batch_slice_coordinates)
        else:
            tuple_of_non_batch_slice_coordinates = (non_batch_slice_coordinates,)

        if self.batch:
            # pick the value for the assignment in each batch row
            batch_rows_coordinate = self.get_batch_rows_coordinate(tuple_of_non_batch_slice_coordinates)
            all_coordinates = (batch_rows_coordinate, *tuple_of_non_batch_slice_coordinates)
        else:
            all_coordinates = tuple_of_non_batch_slice_coordinates

        self.check_all_batch_coordinates_are_1d_and_have_the_same_size(all_coordinates)

        return self.raw_tensor[all_coordinates]

    def get_batch_rows_coordinate(self, tuple_of_non_batch_slice_coordinates):
        # TODO: Using range is much slower than using slice(None)
        # however slice(None) does not produce the same result if there are batch coordinates.
        # For example, assume a 2D n x n tensor A.
        # A[range(n), range(n)] is its diagonal,
        # A[slice(None), range(n)] is equal to A.
        # However, if there are no batch coordinates then we can use slice and obtain the same result,
        # hence the case analysis used here.
        if any(is_batch_coordinate(c) for c in tuple_of_non_batch_slice_coordinates):
            batch_rows_coordinate = range(self.number_of_batch_rows())
        else:
            batch_rows_coordinate = slice(None)
        return batch_rows_coordinate

    def check_all_batch_coordinates_are_1d_and_have_the_same_size(self, all_coordinates):
        batch_coordinates = [c for c in all_coordinates if is_batch_coordinate(c)]

        invalid_batch_coordinate = first(batch_coordinates, lambda bc: len(bc) != 0 and is_iterable(bc[0]))
        if invalid_batch_coordinate is not None:
            raise BatchCoordinateFirstElementIsIterable(invalid_batch_coordinate)

        set_of_len_of_batch_coordinates = {len(batch_coordinate) for batch_coordinate in batch_coordinates}

        if len(set_of_len_of_batch_coordinates) > 1:
            raise BatchCoordinatesDoNotAgreeException()

    def slice(self, non_batch_slice_coordinates):
        raw_tensor_slice = self.get_raw_tensor_slice(non_batch_slice_coordinates)  # already covers batch cases
        new_table_is_batch = self.batch or contains_batch_coordinate(non_batch_slice_coordinates)
        return self.new_table_from_raw_entries(raw_tensor_slice, new_table_is_batch)

    # Methods depending on structure and value representation choice (here, normal space rather than log)

    # This distinction is useful if a sub-class represents values in some other system
    # (such as in log-space), because then we know these methods do need to be overridden.

    def __getitem__(self, non_batch_slice_coordinates):
        """Does what super method does, with the additional fact that the returned value is a torch.Tensor"""
        return self.get_raw_tensor_slice(non_batch_slice_coordinates)

    def __mul__(self, other):
        """
        Multiply by another table, taking care of batch concerns.
        Delegates actually multiplication of raw tensors
        so that sub-classes can reuse this method and override that method only.
        """
        assert self.non_batch_shape == other.non_batch_shape, f"Multiplied {type(self)} instances must have the same non-batch shapes, but have {self.non_batch_shape} and {other.non_batch_shape}"
        assert not (
                    self.batch and other.batch) or self.number_of_batch_rows() == other.number_of_batch_rows(), f"If both tables are batches then they must have the same number of rows, but they have {self.number_of_batch_rows()} and {other.number_of_batch_rows()} rows respectively"

        result_is_batch = self.batch or other.batch

        if type(self) == type(other):
            # if 'other' is the same type as self, more assumptions can be made about raw tensors,
            # allowing greater efficiency -- the assumptions are encapsulated
            # in the implementation of method raw_tensor_of_product_of_potentials_of_raw_tensors
            raw_tensor = self.raw_tensor_of_product_of_potentials_of_raw_tensors(self.raw_tensor, other.raw_tensor)
            return self.new_table_from_raw_entries(raw_tensor, result_is_batch)
        else:
            # otherwise, pay the penalty of possibly converting back and forth from non-potential raw tensors.
            # Note: operating on potentials is not less efficiency for class PyTorchTable,
            # but is be for sub-classes using different representations, such as PyTorchLogTable
            array_of_potentials = self.potentials_tensor() * other.potentials_tensor()
            return self.from_array(array_of_potentials, result_is_batch)

    @staticmethod
    def raw_tensor_of_product_of_potentials_of_raw_tensors(raw_tensor_1, raw_tensor_2):
        """
        Returns the raw_tensor representing the product of potentials corresponding to two raw tensors
        (raw tensors must come from the same class).
        """
        return raw_tensor_1 * raw_tensor_2

    def sum_out(self, dim):
        if util.is_empty_iterable(dim) and self.batch:
            # PyTorch treats an empty sequence of dims to mean "all dims"
            # If dim is an empty list, so will be effective_dim.
            # If we have a batch, using torch.sum on the raw tensor with empty dim will sum everything, including
            # the batch rows, when we actually want the sum to operate over all non-batch dims only but preserve
            # the batch dimension.
            # So we must replace dim by [0,...,len(non_batch_shape) - 1]
            # (later this will be transformed into effective_dim that takes the batch dimension into account)
            dim = list(range(len(self.non_batch_shape)))

        if self.batch:
            def effective_dimension_value(d):
                return d + 1
        else:
            def effective_dimension_value(d):
                return d

        effective_dim = util.map_iterable_or_value(effective_dimension_value, dim)

        potential_tensor = torch.sum(self.raw_tensor, effective_dim)
        return self.from_array(potential_tensor, self.batch)

    def sum(self):
        if self.batch:
            return torch.sum(self.raw_tensor, dim=all_dims_but_first(self.raw_tensor))
        else:
            return torch.sum(self.raw_tensor)

    def argmax(self):
        indices = batch_argmax(self.raw_tensor, batch_dim=1 if self.batch else 0)
        return indices

    def normalize(self):
        if self.batch:
            # we want to divide each row by its sum, which is provided as a one-dimensional tensor by self.sum()
            # To get the proper broadcasting to work, we must reshape it with trailing dimensions of size 1
            # for the non-batch dimensions.
            trailing_size_one_dimensions = [1] * len(self.non_batch_shape)
            n_rows = self.raw_tensor.shape[0]
            partitions = self.sum().reshape(n_rows, *trailing_size_one_dimensions)
            normalized_potential_tensor = self.raw_tensor / partitions
            return self.from_array(normalized_potential_tensor, self.batch)
        else:
            normalized_potential_tensor = self.raw_tensor / self.sum()
            return self.from_array(normalized_potential_tensor)

    def sample(self):
        batch_size = self.number_of_batch_rows() if self.batch else 1
        non_batch_size = math.prod(self.non_batch_shape)
        batch_of_potentials_in_rows = self.potentials_tensor().reshape(batch_size, non_batch_size)
        sampled_non_batch_assignment_indices = Categorical(batch_of_potentials_in_rows).sample()
        assignments = self.from_tensor_of_non_batch_assignment_indices_to_tensor_of_non_batch_assignments(
            sampled_non_batch_assignment_indices)
        return assignments

    def from_tensor_of_non_batch_assignment_indices_to_tensor_of_non_batch_assignments(
            self, non_batch_assignment_indices):
        """Takes a 1D tensor of non-batch assignment indices and returns the corresponding assignments"""
        return self.non_batch_radices.representation(non_batch_assignment_indices)

    @property
    def non_batch_radices(self):
        if self._non_batch_radices is None:
            self._non_batch_radices = TensorMixedRadix(self.non_batch_shape)
        return self._non_batch_radices

    def randomize(self):
        self.raw_tensor = torch.rand(self.raw_tensor.shape, requires_grad=True)

    def randomized_copy(self):
        potential_tensor = torch.rand(self.raw_tensor.shape, requires_grad=True)
        return self.from_array(potential_tensor, self.batch)

    def __eq__(self, other):
        """
        Indicates whether two tables are equal, including whether they are batches or not and, if so,
        of the same size. Once that is checked, their potentials are compared with
        torch.allclose with atol=1e-3 and rtol=1e-3
        """
        if isinstance(other, PyTorchTable):
            result = self.has_same_batch_and_shape(other) and \
                     torch.allclose(self.potentials_tensor(), other.potentials_tensor(), atol=1e-3, rtol=1e-3)
            return result
        else:
            raise Exception("Comparison of PyTorchTable to tables other than PyTorchTable not implemented")

    def has_same_batch_and_shape(self, other):
        return self.number_of_batch_rows() == other.number_of_batch_rows() and \
               self.non_batch_shape == other.non_batch_shape

    def __repr__(self):
        """
        Prints a representation for the current table.
        The method uses method 'potentials_tensor' so sub-classes only need to override that to re-use this method.
        """
        return \
            ("batch " if self.batch else "") + \
            str(map_of_nested_list(lambda v: round(v, 4), self.potentials_tensor().tolist()))

    def potentials_tensor(self):
        return self.raw_tensor


class BatchCoordinateFirstElementIsIterable(Exception):
    """
    Exception throw when indexing the table with a batch coordinate whose first element is an iterable.
    This catches mistakes in which the user sends a batch coordinate not containing scalars.
    Only the first element is checked for efficiency (checking all elements would take much longer and
    be unnecessary because if some elements and iterable and others are not a PyTorch exception
    will be raised anyway).
    """

    def __init__(self, invalid_batch_coordinate):
        super(BatchCoordinateFirstElementIsIterable, self).__init__(
            f"Batch coordinate first element is iterable; they should all be integral scalars: "
            f"{invalid_batch_coordinate[0]}")


class BatchCoordinatesDoNotAgreeException(Exception):
    """
    Exception throw when indexing the table with batch coordinates (that is, coordinates with more than one value
    for batch processing) are not all of the same type or not agreeing with a batch table's number of row (
    in the case the table is a batch).
    """

    def __init__(self):
        super(BatchCoordinatesDoNotAgreeException, self).__init__(
            f"Batch coordinates do not agree with each other or with number of rows in batch table.")
