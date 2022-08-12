import sympy

from neuralpp.symbolic.basic_expression import BasicExpression
from neuralpp.symbolic.expression import Variable
from neuralpp.symbolic.basic_expression import BasicIntegral, BasicVariable
from neuralpp.symbolic.z3_expression import Z3SolverExpression
from neuralpp.symbolic.polynomial_approximation import get_normal_piecewise_polynomial_approximation
from neuralpp.symbolic.lazy_normalizer import LazyNormalizer
from neuralpp.symbolic.sympy_expression import SymPyExpression
from fractions import Fraction
from sympy.utilities.autowrap import autowrap
from timeit import timeit
from pickle import dump, load
import time

x = BasicVariable("x", float)
mu1 = BasicVariable("mu1", float)
mu2 = BasicVariable("mu2", float)

# P(x, mu1, mu2) = Normal(x | mu1, 1.0) * Normal(mu1 | mu2, 1.0) * Normal(mu2 | 0.0, 1.0) propto
formula = get_normal_piecewise_polynomial_approximation(x, mu1, 1.0)
joint1 = get_normal_piecewise_polynomial_approximation(x, mu1, 1.0) \
        * \
        get_normal_piecewise_polynomial_approximation(mu1, mu2, 1.0) \
        * \
        get_normal_piecewise_polynomial_approximation(mu2, BasicExpression.new_constant(0.0), 1.0)
simple_polynomials = (x ** 3 + x ** 2 + x ** 1) * (mu1 ** 3 + mu1 ** 2 + mu1 ** 1)

joint_simple = get_normal_piecewise_polynomial_approximation(x, mu1, 1.0) \
               * \
               get_normal_piecewise_polynomial_approximation(mu1, BasicExpression.new_constant(0.0), 1.0) \

#  if A & B then C else D
# if A then if B then C else D


def evaluation0():
    start = time.time()
    goal = BasicIntegral(mu1, Z3SolverExpression.from_expression(mu1 > -20.0) & (mu1 < 20.0), formula)
    result = LazyNormalizer.normalize(goal, Z3SolverExpression())
    end = time.time()
    print(f"evaluation result: {result}")
    sympy_formula = SymPyExpression.convert(result).sympy_object
    print(f"evaluation result: {sympy_formula}")

    print(f"in time {end - start}")
    sympy_formula_cython = autowrap(sympy_formula, backend='cython', tempdir='../../../../autowraptmp')
    print("sympy_formula.subs")
    xx = sympy.symbols('x')
    print(sympy_formula.subs({xx: 1.0}))
    print("sympy_formula_cython")
    print(sympy_formula_cython(1.0))
    print(timeit(lambda: sympy_formula_cython(1.0), number=1000))


def evaluation():
    start = time.time()
    goal = BasicIntegral(mu1, Z3SolverExpression.from_expression((mu1 > -20.0) & (mu1 < 20.0)), joint1)
    result = LazyNormalizer.normalize(goal, Z3SolverExpression())
    end = time.time()
    print(f"evaluation result: {result}")
    sympy_formula = SymPyExpression.convert(result).sympy_object
    print(f"evaluation result: {sympy_formula}")

    print(f"in time {end - start}")

    sympy_formula_cython = autowrap(sympy_formula, backend='cython', tempdir='../../../../autowraptmp')
    print("sympy_formula.subs")
    xx, mu2mu2 = sympy.symbols('x mu2')
    print(sympy_formula.subs({xx: 1.0, mu2mu2: 0.0}))
    print("sympy_formula_cython")
    print(sympy_formula_cython(1.0, 0.0))
    print(timeit(lambda: sympy_formula_cython(1.0), number=1000))


def evaluation_simple():
    start = time.time()
    goal = BasicIntegral(mu1, Z3SolverExpression.from_expression(mu1 > -20.0) & (mu1 < 20.0), joint_simple)
    result = LazyNormalizer.normalize(goal, Z3SolverExpression())
    end = time.time()
    print(f"evaluation result: {result}")
    sympy_formula = SymPyExpression.convert(result).sympy_object
    print(f"evaluation result: {sympy_formula}")

    print(f"in time {end - start}")
    sympy_formula_cython = autowrap(sympy_formula, backend='cython', tempdir='../../../../autowraptmp')
    print("sympy_formula.subs")
    xx = sympy.symbols('x')
    print(sympy_formula.subs({xx: 1.0}))
    print("sympy_formula_cython")
    print(sympy_formula_cython(1.0))
    print(timeit(lambda: sympy_formula_cython(1.0), number=1000))


def evaluation_simple():
    start = time.time()
    goal = BasicIntegral(mu1, Z3SolverExpression.from_expression(mu1 > -20.0) & (mu1 < 20.0), joint_simple)
    result = LazyNormalizer.normalize(goal, Z3SolverExpression())


if __name__ == "__main__":
    # evaluation0()
    # evaluation_simple()
    evaluation()


# P(x, mu2) propto
# phi_x_mu2 = sum_mu1 joint   # we use phi to indicate an unnormalized distribution

# P(x) propto
# phi_x = sum_mu1 phi_x_mu2

# P(mu2 | x) = phi_x_mu2 / phi_x

# This is symbolic in mu2 and x. To obtain the posterior for mu2 for a given evidence (observation) x,
# simply plug it and evaluate.


