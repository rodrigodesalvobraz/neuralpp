from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)


class FixedPyTorchTableFactor(PyTorchTableFactor):

    # Main functionality: overriding pytorch_parameters so empty list so it does not get changed during training

    def pytorch_parameters(self):
        return []

    # Overridden construction

    def __init__(
        self, variables, array_or_table_of_potentials, log_space=True, batch=False
    ):
        super().__init__(
            variables, array_or_table_of_potentials, log_space=log_space, batch=batch
        )

    @staticmethod
    def from_function(variables, function, log_space=True, batch_size=None):
        return PyTorchTableFactor.from_function_and_constructor(
            variables,
            function,
            FixedPyTorchTableFactor,
            log_space=log_space,
            batch_size=batch_size,
        )

    @staticmethod
    def from_predicate(variables, predicate, log_space=True, batch_size=None):
        return FixedPyTorchTableFactor.from_function(
            variables, lambda *args: float(predicate(*args)), log_space, batch_size
        )
