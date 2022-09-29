from typing import Callable, Iterable, Sized

import numpy as np
from sympy import Poly
import torch.distributions
from torch.distributions import Normal

from neuralpp.symbolic.basic_expression import BasicExpression
from neuralpp.symbolic.sympy_expression import make_piecewise, SymPyExpression
from neuralpp.symbolic.expression import Expression, Variable, Constant
from neuralpp.util.util import pairwise
from neuralpp.symbolic.sympy_interpreter import SymPyInterpreter

simplifer = SymPyInterpreter()


def get_normal_piecewise_polynomial_approximation(
    variable: Variable, mean: Expression, sigma: float, generator: Variable
) -> Expression:
    """
    @param generator: SymPy term for "variable" in a polynomial (see https://docs.sympy.org/latest/modules/polys/basics.html)
    Returns an Expression equivalent to a polynomial that approximates the density of Normal(mean, sigma).
    """
    sigma_constant = BasicExpression.new_constant(sigma)
    if sigma == 1:
        if isinstance(mean, Constant) and mean.value == 0:
            new_var = variable
        else:
            new_var = simplifer.simplify(expression=variable - mean)
    else:
        new_var = simplifer.simplify(expression=(variable - mean) / sigma_constant)

    standard_normal_piecewise_polynomial = (
        get_standard_normal_piecewise_polynomial_approximation(new_var, generator)
    )

    if sigma == 1:
        return standard_normal_piecewise_polynomial
    else:
        return 1 / sigma_constant * standard_normal_piecewise_polynomial


def get_standard_normal_piecewise_polynomial_approximation(
    variable: Variable, generator: Variable
) -> Expression:
    """
    Returns an Expression equivalent to a polynomial that approximates the density of a standard Normal distribution.
    """
    std_normal = Normal(0.0, 1.0)
    f = lambda x: std_normal.log_prob(torch.tensor(x)).exp()
    segment_boundaries = [float(v) for v in [-4, -3.25, -1.75, 0, 1.75, 3.25, 4]]
    segment_degrees = [3] * len(segment_boundaries)
    return piecewise_polynomial_approximation(
        f, variable, segment_degrees, segment_boundaries, generator
    )


def piecewise_polynomial_approximation(
    f: Callable[[float], float],
    variable: Variable,
    degree_per_segment: Iterable[int],
    segment_boundaries: Iterable[float],
    generator: Variable,
) -> Expression:
    """
    Returns an Expression representing a piecewise polynomial approximating f(variable) in the following way:
    approximation(variable) =
        0, if variable < segment_boundaries[0] or variable > segment_boundaries[-1]
        least-squares polynomial approximation to f(variable) in [segment_boundary[i], segment_boundary[i+1]],
             with degree degree_per_segment[i] if variable in [segment_boundary[i], segment_boundary[i+1]].
    """
    polynomials = [
        polynomial_approximation(f, variable, start, end, degree, generator)
        for ((start, end), degree) in zip(
            pairwise(segment_boundaries), degree_per_segment
        )
    ]

    conditions = [
        (start <= variable) & (variable < end)
        for (start, end) in pairwise(segment_boundaries)
    ]

    piecewise_polynomial = make_piecewise_expression(conditions, polynomials)

    return piecewise_polynomial


def polynomial_approximation(
    f: Callable[[float], float],
    variable: Expression,
    start: float,
    end: float,
    degree: int,
    generator: Variable,
) -> Expression:
    """
    Returns an Expression that is equivalent to a polynomial with the specified degree in variable
    approximating f(variable) in the [start, end] interval.
    """
    xs = np.linspace(start, end, degree + 1)
    assert len(xs) == degree + 1
    ys = np.array([f(x) for x in xs])
    coefficients = np.polyfit(xs, ys, degree)
    polynomial = from_coefficients_to_polynomial(variable, coefficients, generator)
    return polynomial


def from_coefficients_to_polynomial(
    variable: Expression, coefficients: Sized, generator: Variable
) -> Expression:
    """
    Given a variable and a sequence of coefficients a0, ..., an,
    returns an Expression representing a0 * variable ** n + a1 * variable ** (n - 1) + ... + an.
    """
    var = SymPyExpression.convert(variable).sympy_object
    gen = SymPyExpression.convert(generator).sympy_object
    var_poly = Poly.from_list(coefficients, var)
    sympy_result = Poly.from_poly(var_poly, gen)
    result = SymPyExpression.from_sympy_object(sympy_result, {var: float, gen: float})
    assert result is not None
    return result


def make_piecewise_expression(conditions, expressions):
    """
    Given conditions C1, ..., Cn and expressions E1, ..., En,
    returns Expression if C1 then E1 else if C2 then E2 else ... else 0.
    assume C1, .., Cn are mutually exclusive
    """
    arguments = [
        arg
        for condition, expression in zip(conditions, expressions)
        for arg in (expression, condition)
    ]
    return make_piecewise(arguments)
