from neuralpp.inference.graphical_model.representation.factor.neural.neural_factor import (
    NeuralFactor,
)
from neuralpp.inference.graphical_model.variable.discrete_variable import (
    DiscreteVariable,
)
from neuralpp.inference.neural_net.MLP import MLP
from neuralpp.util import util


class MLPFactor(NeuralFactor):
    """A convenience class for a NeuralFactor using an MLP network"""

    @staticmethod
    def make(input_variables, *hidden_layer_sizes_and_output_variable):
        hidden_layer_sizes, output_variable = MLPFactor.validate_arguments(
            input_variables, hidden_layer_sizes_and_output_variable
        )
        neural_net = MLPFactor.make_neural_net(
            input_variables, hidden_layer_sizes, output_variable
        )
        return MLPFactor(neural_net, input_variables, output_variable)

    @staticmethod
    def make_neural_net(input_variables, hidden_layer_sizes, output_variable):
        all_layer_sizes = MLPFactor.compute_mlp_all_layer_sizes_list(
            input_variables, hidden_layer_sizes, output_variable
        )
        neural_net = MLP(*all_layer_sizes)
        return neural_net

    @staticmethod
    def compute_mlp_all_layer_sizes_list(
        input_variables, hidden_layer_sizes, output_variable
    ):
        first_layer_size = len(input_variables)
        last_layer_size = output_variable.cardinality
        all_layer_sizes = [first_layer_size, *hidden_layer_sizes, last_layer_size]
        return all_layer_sizes

    @staticmethod
    def validate_arguments(input_variables, args):
        try:
            hidden_layer_sizes = args[:-1]
            output_variable = args[-1]
            assert (
                all(isinstance(v, DiscreteVariable) for v in input_variables)
                and all(isinstance(i, int) for i in hidden_layer_sizes)
                and isinstance(output_variable, DiscreteVariable)
            )
        except Exception:
            raise Exception(
                f"MLPFactor constructor must receive a list of input variables (all of type {DiscreteVariable.__name__}), "
                f"the sizes of hidden layer (each an argument of the constructor as opposed to being provided in"
                f"a sequence), "
                f"and an output variable (all also of type {DiscreteVariable.__name__}) but got instead "
                f"{input_variables} and {util.join(args)}"
            )
        return hidden_layer_sizes, output_variable
