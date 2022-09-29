import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.distributions import Normal

from neuralpp.symbolic.basic_expression import BasicExpression
from neuralpp.symbolic.basic_interpreter import BasicInterpreter
from neuralpp.symbolic.polynomial_approximation import (
    get_normal_piecewise_polynomial_approximation,
)

plot = False


def test_normal_polynomial_approximation():
    print("Computing approximation")
    mean_value = 3.0
    sigma_value = 2.0
    mean = BasicExpression.new_constant(mean_value)
    x = BasicExpression.new_variable("x", float)
    normal_piecewise_polynomial_approximation = (
        get_normal_piecewise_polynomial_approximation(x, mean, sigma_value, x)
    )

    x_axis = np.linspace(
        -6 * sigma_value + mean_value, 6 * sigma_value + mean_value, 800
    )

    print("Computing true values")
    true_normal_values = [
        Normal(mean_value, sigma_value).log_prob(torch.tensor(x)).exp()
        for x in x_axis
    ]

    interpreter = BasicInterpreter()

    print("Computing approximate values")

    def approximate_normal(value: float) -> float:
        return interpreter.eval(
            normal_piecewise_polynomial_approximation.replace(
                x, BasicExpression.new_constant(value)
            )
        )

    # approximate_normal_values = [approximate_normal(x) for x in x_axis]

    # epsilon = 0.01 / sigma_value
    # assert all([abs(t - a) < epsilon for (t, a) in zip(true_normal_values, approximate_normal_values)])

    # if plot:
    #     print("Plotting")
    #     plt.plot(x_axis, true_normal_values, label="Gaussian")
    #     plt.plot(x_axis, approximate_normal_values, label="Approximation")
    #     plt.legend(loc="upper left")
    #     plt.show()
