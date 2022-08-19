from typing import Callable, Iterable, Sized

import numpy as np
import torch.distributions
from torch.distributions import Normal

from neuralpp.symbolic.basic_expression import BasicExpression
from neuralpp.symbolic.constants import if_then_else
from neuralpp.symbolic.expression import Expression, Variable
from neuralpp.util.util import pairwise


def get_normal_piecewise_polynomial_approximation(variable: Variable, mean: Expression, sigma: float) -> Expression:
    """
    Returns an Expression equivalent to a polynomial that approximates the density of Normal(mean, sigma).
    """
    standard_normal_piecewise_polynomial = get_standard_normal_piecewise_polynomial_approximation(variable)
    sigma_constant = BasicExpression.new_constant(sigma)
    return 1/sigma_constant * standard_normal_piecewise_polynomial.replace(variable, (variable - mean)/sigma_constant)


def get_standard_normal_piecewise_polynomial_approximation(variable: Variable) -> Expression:
    """
    Returns an Expression equivalent to a polynomial that approximates the density of a standard Normal distribution.
    """
    std_normal = Normal(0.0, 1.0)
    f = lambda x: std_normal.log_prob(torch.tensor(x)).exp()
    segment_boundaries = [float(v) for v in [-4, -3.25, -1.75, 0, 1.75, 3.25, 4]]
    segment_degrees = [3] * len(segment_boundaries)
    return piecewise_polynomial_approximation(f, variable, segment_degrees, segment_boundaries)


def piecewise_polynomial_approximation(
        f: Callable[[float], float],
        variable: Variable,
        degree_per_segment: Iterable[int],
        segment_boundaries: Iterable[float]) -> Expression:
    """
    Returns an Expression representing a piecewise polynomial approximating f(variable) in the following way:
    approximation(variable) =
        0, if variable < segment_boundaries[0] or variable > segment_boundaries[-1]
        least-squares polynomial approximation to f(variable) in [segment_boundary[i], segment_boundary[i+1]],
             with degree degree_per_segment[i] if variable in [segment_boundary[i], segment_boundary[i+1]].
    """
    polynomials = [
        polynomial_approximation(f, variable, start, end, degree)
        for ((start, end), degree)
        in zip(pairwise(segment_boundaries), degree_per_segment)
    ]

    conditions = [(start <= variable) & (variable < end) for (start, end) in pairwise(segment_boundaries)]

    piecewise_polynomial = make_piecewise_expression(conditions, polynomials)

    return piecewise_polynomial


def polynomial_approximation(
        f: Callable[[float], float],
        variable: Expression,
        start: float,
        end: float,
        degree: int) -> Expression:
    """
    Returns an Expression that is equivalent to a polynomial with the specified degree in variable
    approximating f(variable) in the [start, end] interval.
    """
    xs = np.linspace(start, end, degree + 1)
    assert len(xs) == degree + 1
    ys = np.array([f(x) for x in xs])
    coefficients = np.polyfit(xs, ys, degree)
    polynomial = from_coefficients_to_polynomial(variable, coefficients)
    return polynomial


def from_coefficients_to_polynomial(variable: Expression, coefficients: Sized) -> Expression:
    """
    Given a variable and a sequence of coefficients a0, ..., an,
    returns an Expression representing a0 * variable ** n + a1 * variable ** (n - 1) + ... + an.
    """
    degree = len(coefficients) - 1
    polynomial = BasicExpression.new_constant(0.)
    for i in range(degree):
        polynomial += float(coefficients[i]) * variable ** BasicExpression.new_constant(degree - i)
    polynomial += float(coefficients[degree])
    return polynomial


def make_piecewise_expression(conditions, expressions):
    """
    Given conditions C1, ..., Cn and expressions E1, ..., En,
    returns Expression if C1 then E1 else if C2 then E2 else ... else 0.
    assume C1, .., Cn are mutually exclusive
    """
    from .sympy_expression import make_piecewise
    return make_piecewise(conditions, expressions)
