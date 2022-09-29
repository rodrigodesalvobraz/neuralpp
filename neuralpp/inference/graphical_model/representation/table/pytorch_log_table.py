import torch
from neuralpp.inference.graphical_model.representation.table.pytorch_table import (
    PyTorchTable,
)
from neuralpp.util.log_util import (
    log_of_nested_list_without_inf_non_differentiable,
    log_without_inf_non_differentiable,
)


class PyTorchLogTable(PyTorchTable):
    def __init__(self, raw_entries, batch=False):
        super().__init__(raw_entries, batch)
        assert not torch.flatten(torch.isnan(self.raw_tensor)).any()

    @staticmethod
    def from_array(array_of_potentials, batch=False):
        if isinstance(array_of_potentials, torch.Tensor):
            tensor = log_without_inf_non_differentiable(array_of_potentials)
        else:
            log_of_potentials = log_of_nested_list_without_inf_non_differentiable(
                array_of_potentials
            )
            tensor = torch.tensor(log_of_potentials, requires_grad=True)
        table = PyTorchLogTable(tensor, batch)
        return table

    @staticmethod
    def from_function(
        shape,
        function_arguments_iterables,
        function_of_potentials,
        batch=False,
    ):
        prob_space = PyTorchTable.from_function(
            shape, function_arguments_iterables, function_of_potentials, batch
        )
        log_probability_tensor = log_without_inf_non_differentiable(
            prob_space.raw_tensor
        )
        leaf_tensor = log_probability_tensor.clone().detach().requires_grad_(True)
        return PyTorchLogTable(leaf_tensor, batch)

    # Methods depending on structure and value representation choice (here, log)

    def __getitem__(self, non_batch_slice_coordinates):
        return super().__getitem__(non_batch_slice_coordinates).exp()

    @staticmethod
    def raw_tensor_of_product_of_potentials_of_raw_tensors(raw_tensor_1, raw_tensor_2):
        return raw_tensor_1 + raw_tensor_2  # + since we are at log-space

    def sum_out(self, dim):

        # We resort to PyTorchTable so we can rely on its ability to manage batches
        normal_space_table = PyTorchTable(self.raw_tensor.exp(), self.batch)
        normal_summed_out = normal_space_table.sum_out(dim)

        # We can use log directly here because the potentials are guaranteed not to be exactly 0
        # since they are sums of potentials coming from neuralpp.exponentiations.
        summed_out_log_raw_tensor = normal_summed_out.raw_tensor.log()

        return PyTorchLogTable(summed_out_log_raw_tensor, self.batch)

    def sum(self):
        normal_space_table = PyTorchTable(self.raw_tensor.exp(), self.batch)
        return normal_space_table.sum()

    def normalize(self):
        normal_space_table = PyTorchTable(self.raw_tensor.exp(), self.batch)
        normalized_in_normal_space = normal_space_table.normalize()
        normalized_in_normal_space_raw_tensor = normalized_in_normal_space.raw_tensor
        # we can use log directly because the normal-space raw tensor came from neuralpp.exponentiation
        normalized_in_log_space_raw_tensor = normalized_in_normal_space_raw_tensor.log()
        return PyTorchLogTable(normalized_in_log_space_raw_tensor, self.batch)

    def randomize(self):
        self.raw_tensor = self.make_random_raw_tensor()

    def randomized_copy(self):
        leaf_log_tensor = self.make_random_raw_tensor()
        return PyTorchLogTable(leaf_log_tensor, self.batch)

    def make_random_raw_tensor(self):
        random_uniform_tensor = torch.rand(self.raw_tensor.shape)
        random_log_uniform_tensor = log_without_inf_non_differentiable(
            random_uniform_tensor
        )
        leaf_log_tensor = (
            random_log_uniform_tensor.clone().detach().requires_grad_(True)
        )
        return leaf_log_tensor

    def potentials_tensor(self):
        return self.raw_tensor.exp()
