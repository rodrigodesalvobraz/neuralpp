class Table:

    def __init__(self):
        self._batch = None
        self._non_batch_shape = None

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, value):
        self._batch = value

    @property
    def non_batch_shape(self):
        return self._non_batch_shape

    @non_batch_shape.setter
    def non_batch_shape(self, value):
        self._non_batch_shape = value

    def __getitem__(self, non_batch_slice_coordinates):
        """
        Returns a tensor corresponding to the values indexed by given non-batch coordinates,
        that is, coordinates for dimensions that are not the batch row index dimension.
        A coordinate can be either an integer, slice(None), or an array-like object of integers
        (TODO: the latter are called 'batch coordinates' in the sense that they represent a batch of values,
        but note that this is confusing because a non-batch coordinate, that is, a coordinate on a non-batch dimension,
        may be a batch coordinate! Sorry for the confusing terminology -- this should be cleaned up).
        To describe the result, let us divide into cases.
        If there are no batch coordinates (that is, no non-batch coordinates with multiple values),
        and the table is not a batch,
        then the result is table[non_batch_slice_coordinates].
        This will be either a scalar, if there are no slice(None) coordinates, or a tensor-like object if there
        are slice(None) coordinates.
        If there are no batch coordinates and the table is a batch, then the result is
        M such that M_i is table[i, *non_batch_slice_coordinates],
        that is, for each row of the table batch, we apply the given non-batch coordinates.
        Note that this will be 1D is there were no slice(None) coordinates, and of greater dimensions otherwise.
        If there are batch coordinates and the table is not a batch, then the result is
        M such that M_i is table[*th(i, non_batch_slice_coordinates)] where
        th is a function that derives a tuple of values from non_batch_slice_coordinates using only the i-th
        element of each batch-coordinate (slice(None) coordinates are left alone).
        Note that this requires all batch coordinates to contain the same number of values.
        Finally, if table is a batch and there are batch coordinates, then the result is
        M such that M_i is table[i, *th(i, non_batch_slice_coordinates)], that is,
        we apply the same procedure for each row i.
        """
        self._not_implemented("__getitem__")

    def __len__(self):
        self._not_implemented("__len__")

    def assignments(self):
        self._not_implemented("assignments")

    def __mul__(self, item):
        self._not_implemented("__mul__")

    def expand(self, shape, dim):
        self._not_implemented("expand")

    def permute(self, permutation):
        self._not_implemented("permute")

    def sum_out(self, dim):
        self._not_implemented("sum_out")

    def sum(self):
        self._not_implemented("sum")

    def argmax(self):
        """
        Returns index arrays (or scalar if there is a single non-batch dimension) for the largest value in the table.
        If the table is a batch, the indices will be a batch of such index arrays or scalars.
        """
        self._not_implemented("argmax")

    def normalize(self):
        self._not_implemented("normalize")

    def sample(self):
        self._not_implemented("single_sample")

    def slice(self, non_batch_slice_coordinates):
        """
        Returns a table corresponding to the data returned from __getitem__ (see its documentation).
        The resulting table will be a batch if it contains 0 or more than 1 rows, and not a batch otherwise.
        """
        self._not_implemented("condition")

    def randomize(self):
        self._not_implemented("randomize")

    def randomized_copy(self):
        self._not_implemented("randomized_copy")

    def _not_implemented(self, name):
        # creating a variable first prevents compiler from thinking this is an abstract method
        error = NotImplementedError(f"{name} not implemented for {type(self)}")
        raise error
