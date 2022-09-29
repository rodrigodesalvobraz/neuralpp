import sympy

from neuralpp.symbolic.basic_expression import BasicExpression
from neuralpp.symbolic.basic_expression import (
    basic_integral,
    BasicVariable,
    BasicConstant,
)
from neuralpp.symbolic.z3_expression import (
    Z3SolverExpression,
    Z3SolverExpressionDummy,
)
from neuralpp.symbolic.polynomial_approximation import (
    get_normal_piecewise_polynomial_approximation,
    make_piecewise_expression,
)
from neuralpp.symbolic.lazy_normalizer import LazyNormalizer
from sympy.utilities.autowrap import autowrap
from typing import Dict, Any
import time

x = BasicVariable("x", float)
mu1 = BasicVariable("mu1", float)
mu2 = BasicVariable("mu2", float)

# P(x, mu1, mu2) = Normal(x | mu1, 1.0) * Normal(mu1 | mu2, 1.0) * Normal(mu2 | 0.0, 1.0) propto
formula = get_normal_piecewise_polynomial_approximation(
    x, mu1, 1.0, generator=mu1
)
joint_simple = get_normal_piecewise_polynomial_approximation(
    x, mu1, 1.0, generator=mu1
) * get_normal_piecewise_polynomial_approximation(
    mu1, BasicExpression.new_constant(0.0), 1.0, generator=mu1
)
joint = (
    get_normal_piecewise_polynomial_approximation(x, mu1, 1.0, generator=mu1)
    * get_normal_piecewise_polynomial_approximation(
        mu1, mu2, 1.0, generator=mu1
    )
    * get_normal_piecewise_polynomial_approximation(
        mu2, BasicExpression.new_constant(0.0), 1.0, generator=mu1
    )
)

E1 = x < 0
E2 = x >= 0
E3 = mu1 < 0
E4 = mu1 >= 0
two_piecewise = make_piecewise_expression(
    [E1, E2], [x**2, x]
) * make_piecewise_expression([E3, E4], [mu1**2, mu1])


def print_piecewise_test():
    from sympy import Piecewise

    a, b = sympy.symbols("a b")
    piecewise_formula = Piecewise((2, a), (3, b))
    sympy_formula_cython = autowrap(
        piecewise_formula, backend="cython", tempdir="../../../../autowraptmp"
    )
    print(sympy_formula_cython(True, False))
    print(sympy_formula_cython(True, True))
    print(sympy_formula_cython(False, True))
    print(sympy_formula_cython(False, False))


def to_printable(sympy_object: sympy.Basic):
    """
    Seems SymPy's codegen cannot print
    """
    if isinstance(sympy_object, sympy.Poly):
        result = sympy_object.as_expr()
        return result
    elif sympy_object.is_Piecewise:
        return sympy_object.func(
            *[(to_printable(expr), cond) for expr, cond in sympy_object.args]
        )
    elif sympy_object.func.is_Add:
        return sympy_object.func(
            *[to_printable(expr) for expr in sympy_object.args]
        )
    else:
        return sympy_object


# def evaluation_general(test_name, goal, variable_value_pairs: Dict[str, Any]):
#     normalizer.profiler.reset()
#     prefix = f"[{test_name}]"
#     start = time.time()
#     result = goal()
#     end = time.time()
#     sympy_obj = SymPyExpression.convert(result).sympy_object
#     sympy_formula = to_printable(sympy_obj)
#     print(sympy_obj)
#     end2 = time.time()
#     print(f"{prefix}evaluation result: {sympy_formula}")
#     print(f"{prefix}in time {end - start:.3g} seconds; convert {end2 - end:.3g} seconds")
#     normalizer.profiler.print_result(prefix)
#
#     sympy_sub_dict = {sympy.symbols(k): v for k, v in variable_value_pairs.items()}
#     python_answer = sympy_formula.subs(sympy_sub_dict)
#     print(f"{prefix}[Python native]sympy.subs answer is {python_answer}")
#     # the following line requires dict to be ordered (supported after Python 3.5)
#     cython_arguments = [v for _, v in variable_value_pairs.items()]
#     sympy_formula_cython = autowrap(sympy_formula, backend='cython', tempdir='../../../../autowraptmp')
#     cython_answer = sympy_formula_cython(*cython_arguments)
#     print(f"{prefix}[Cython]autowrap answer is {cython_answer}")
#     python_run_time = timeit(lambda: sympy_formula.subs(sympy_sub_dict), number=1000)
#     print(f"{prefix}[Python native]run time is {python_run_time * 1000:.3g} microseconds")
#     cython_run_time = timeit(lambda: sympy_formula_cython(*cython_arguments), number=1000)
#     # 1000 times of test run in {cython_run_time} seconds --> 1 time of test run in {cython_run_time} miliseconds
#     print(f"{prefix}[Cython]run time is {cython_run_time * 1000:.3g} microseconds")


def evaluation_general(test_name, goal, variable_value_pairs: Dict[str, Any]):
    normalizer.profiler.reset()
    prefix = f"[{test_name}]"
    start = time.time()
    # result is not "printable", getting to printable would actually take a lot of time.
    result = goal()
    end = time.time()
    print(f"{prefix}in time {end - start:.3g} seconds")
    normalizer.profiler.print_result(prefix)


normalizer = LazyNormalizer()


def normalization_generator(goal):
    """
    @param goal: a Callable that takes no argument and returns the goal
    @return: a callable that returns the normalized result
    we're returning callable so that this can be evaluated lazily
    """
    return lambda: normalizer.normalize(
        goal(), Z3SolverExpressionDummy()
    )  # Z3SolverExpressionDummy is faster when few branch prunings happen
    # result = normalizer.normalize(goal, Z3SolverExpression())


def two_normals_expectation():
    joint_simple_x_eq_0 = joint_simple.replace(x, BasicConstant(0.0))
    P_x_eq_0 = basic_integral(
        mu1,
        Z3SolverExpression.from_expression(mu1 > -20.0) & (mu1 < 20.0),
        joint_simple_x_eq_0,
    )
    P_x_eq_0 = normalizer.normalize(P_x_eq_0, Z3SolverExpressionDummy())
    print(P_x_eq_0)
    P_mu1_given_x_eq_0 = joint_simple_x_eq_0 / P_x_eq_0
    E_mu1 = basic_integral(
        mu1,
        Z3SolverExpression.from_expression(mu1 > -20.0) & (mu1 < 20.0),
        mu1 * P_mu1_given_x_eq_0,
    )
    return normalizer.normalize(E_mu1, Z3SolverExpressionDummy())


if __name__ == "__main__":
    evaluation_general(
        "two piecewise",
        normalization_generator(
            lambda: basic_integral(
                mu2,
                Z3SolverExpression.from_expression(mu2 > -20.0)
                & (mu2 < 20.0),
                two_piecewise,
            )
        ),
        {"x": 10.0, "mu1": 10.0},
    )
    # FIXME: the line below is commented out because currently it raises an exception
    # evaluation_general("2 Normals-expectation", two_normals_expectation, {})
    evaluation_general(
        "1 Normal",
        normalization_generator(
            lambda: basic_integral(
                mu1,
                Z3SolverExpression.from_expression(mu1 > -20.0)
                & (mu1 < 20.0),
                formula,
            )
        ),
        {"x": 0.0},
    )
    evaluation_general(
        "2 Normals-concrete x=1",
        normalization_generator(
            lambda: basic_integral(
                mu1,
                Z3SolverExpression.from_expression(mu1 > -20.0)
                & (mu1 < 20.0),
                joint_simple.replace(x, BasicConstant(1.0)),
            )
        ),
        {},
    )
    evaluation_general(
        "2 Normals-concrete x=0",
        normalization_generator(
            lambda: basic_integral(
                mu1,
                Z3SolverExpression.from_expression(mu1 > -20.0)
                & (mu1 < 20.0),
                joint_simple.replace(x, BasicConstant(0.0)),
            )
        ),
        {},
    )
    evaluation_general(
        "2 Normals",
        normalization_generator(
            lambda: basic_integral(
                mu1,
                Z3SolverExpression.from_expression(mu1 > -20.0)
                & (mu1 < 20.0),
                joint_simple,
            )
        ),
        {"x": 1.0},
    )
    evaluation_general(
        "Joint-concrete x",
        normalization_generator(
            lambda: basic_integral(
                mu1,
                Z3SolverExpression.from_expression(mu1 > -20.0)
                & (mu1 < 20.0),
                joint.replace(x, BasicConstant(0.0)),
            )
        ),
        {"mu2": 0.0},
    )

    # P(x, mu2) propto
    # phi_x_mu2 = sum_mu1 joint   # we use phi to indicate an unnormalized distribution
    #
    # P(x) propto
    # phi_x = sum_mu1 phi_x_mu2
    #
    # P(mu2 | x) = phi_x_mu2 / phi_x
    # This is symbolic in mu2 and x. To obtain the posterior for mu2 for a given evidence (observation) x,
    # simply plug it and evaluate.
    evaluation_general(
        "Joint",
        normalization_generator(
            lambda: basic_integral(
                mu1,
                Z3SolverExpression.from_expression(mu1 > -20.0)
                & (mu1 < 20.0),
                joint,
            )
        ),
        {"x": 0.0, "mu2": 0.0},
    )
