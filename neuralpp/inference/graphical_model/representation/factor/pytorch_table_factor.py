from neuralpp.inference.graphical_model.representation.factor.table_factor import (
    TableFactor,
)
from neuralpp.inference.graphical_model.representation.table.pytorch_log_table import (
    PyTorchLogTable,
)
from neuralpp.inference.graphical_model.representation.table.pytorch_table import (
    PyTorchTable,
)
from neuralpp.inference.graphical_model.representation.table.table_util import (
    shape,
)


class PyTorchTableFactor(TableFactor):
    def __init__(
        self,
        variables,
        array_or_table_of_potentials,
        log_space=True,
        batch=False,
    ):
        super().__init__(
            variables,
            array_or_table_of_potentials
            if isinstance(array_or_table_of_potentials, PyTorchTable)
            else PyTorchLogTable.from_array(array_or_table_of_potentials, batch=batch)
            if log_space
            else PyTorchTable.from_array(array_or_table_of_potentials, batch=batch),
        )
        assert self.table.non_batch_shape == (
            variables_shape := tuple(v.cardinality for v in variables)
        ), (
            f"{type(self).__name__} created with variables {variables} with shape {variables_shape} "
            f"but with table with (non-batch) shape {self.table.non_batch_shape}"
        )

    def pytorch_parameters(self):
        return self.table.pytorch_parameters()

    @staticmethod
    def from_function_and_constructor(
        variables, function, constructor, log_space=True, batch_size=None
    ):
        """
        Builds a PyTorchTableFactor from neuralpp.its variables, a function and a constructor.
        If batch_size is None, creates a non-batch factor.
        If batch_size is an integer, creates a batch factor; in this case, first parameters of function must be
        the batch row index.
        """

        table_class = PyTorchLogTable if log_space else PyTorchTable

        batch = batch_size is not None

        variable_ranges = [range(v.cardinality) for v in variables]
        ranges = variable_ranges if not batch else [range(batch_size)] + variable_ranges

        table_shape = shape(variables) if not batch else (batch_size, *shape(variables))

        table = table_class.from_function(table_shape, ranges, function, batch)

        return constructor(variables, table)

    @staticmethod
    def from_function(variables, function, log_space=True, batch_size=None):
        """
        Builds a PyTorchTableFactor from neuralpp.its variables and a function.
        If batch_size is None, creates a non-batch factor.
        If batch_size is an integer, creates a batch factor; in this case, first parameters of function must be
        the batch row index.
        """
        return PyTorchTableFactor.from_function_and_constructor(
            variables,
            function,
            PyTorchTableFactor,
            log_space=log_space,
            batch_size=batch_size,
        )

    @staticmethod
    def from_predicate(variables, predicate, log_space=True, batch_size=None):
        return PyTorchTableFactor.from_function(
            variables,
            lambda *args: float(predicate(*args)),
            log_space,
            batch_size,
        )
