from neuralpp.inference.graphical_model.representation.factor.neural.neural_factor import (
    NeuralFactor,
)
from neuralpp.inference.graphical_model.variable.integer_variable import (
    IntegerVariable,
)
from neuralpp.inference.neural_net.MLP import MLP
from neuralpp.test.slow_tests.graphical_model.representation.factor.neural.neural_factor_test_util import (
    check_and_show_conditional_distributions,
)
from neuralpp.util.util import try_noisy_test_up_to_n_times


def mlp_neural_factor_generates_distributions():
    x = IntegerVariable("x", 2)
    y = IntegerVariable("y", 4)
    z = IntegerVariable("z", 4)
    input_variables = [x, y]
    output_variable = z
    neural_net = MLP(len(input_variables), 2, 2, output_variable.cardinality)
    neural_factor = NeuralFactor(neural_net, input_variables, output_variable)

    check_and_show_conditional_distributions(neural_factor)


def test_joint_learning():
    try_noisy_test_up_to_n_times(mlp_neural_factor_generates_distributions, n=3)
