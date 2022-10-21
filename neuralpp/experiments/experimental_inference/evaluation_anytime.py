import random
import timeit

from matplotlib import pyplot as plt

from neuralpp.experiments.experimental_inference.approximations import message_approximation
from neuralpp.experiments.experimental_inference.exact_belief_propagation import AnytimeExactBeliefPropagation
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.random.random_model import generate_model
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.graphical_model.variable_elimination import VariableElimination
from neuralpp.util.util import measure_time


def test_monotonic_improvement():
    """
    This test introduces a linear factor graph in which each factor is "informative",
    that is, producing a message approximation that is better than the uniform approximation.
    This should produce a sequence of approximations at the query that is monotonically improving
    as anytime progresses.
    The informative factors are of the form P(xi | x_{i+1}) = 0.9 if x_{i+1} else 0.2,
    with a final variable x_n with a uniform prior. The variables are boolean.
    This model is such that every considered factor either improves the approximation or leaves it the same.
    The error is computed with respect to P(query = true).
    """

    # How to use this method.
    # The method is meant to be a test but it has a few configuration options for interactive experimentation.

    # Beginning of configuration

    show_plot = False  # whether to show a plot of errors as anytime is iterated

    # docstring model (linear graph)
    n = 10  # number of factors in the docstring linear model.
    random_query = False  # whether to pick a query at random for the docstring model or use x_0 instead.

    # random model
    use_random_models = True  # use a random model rather than the one described in the docstring.
    number_of_factors = 25  # number of factors in the random model, if used.
    number_of_variables = int(number_of_factors * 2 / 3)  # number of variables in the random model, if used.

    # End of configuration

    # We pick shallower nodes for expansion first since these are the most informative (breadth-first).
    def scoring_function(node, partial_tree, full_tree):
        return -full_tree.depth(node)

    query = None

    if not use_random_models:
        # Linear models
        monotonic = True
        name = f"Linear model with {n - 1} factors and {n} variables"
        x = [IntegerVariable(f"x{i}", 2) for i in range(n)]
        if not random_query:
            query = x[0]
        factors = [
            PyTorchTableFactor.from_function(
                [x[i], x[i + 1]],
                lambda xi, xip1:
                (0.9 if xi else 0.1) if xip1 else
                (0.2 if xi else 0.8))
            for i in range(n - 1)]
    else:
        # Random models
        name = f"Random model with {number_of_factors} factors and {number_of_variables} variables"
        monotonic = False
        factors = generate_model(
            number_of_factors=number_of_factors, number_of_variables=number_of_variables, cardinality=2
        )

    if query is None:
        query = random.choice([v for f in factors for v in f.variables])

    evaluate_anytime(factors, query)


def test_deep_model():
    def generate_deep_model(variable_count: int, min_depth: int, branches: int, cross_edges: int):
        if variable_count - 1 < branches * (min_depth - 1):
            raise Exception(
                f"Cannot produce a model of depth ${min_depth}, ${variable_count} variables, and ${branches} branches")
        x = [IntegerVariable(f"x{i}", 2) for i in range(1, variable_count)]
        root = IntegerVariable("x0", 2)
        model = []
        depth = {0: 0}
        edges = set([])

        def create_bivariate_factor(i, j):
            depth[j] = depth[i] + 1
            prob1 = random.uniform(0.0, 1.0)
            prob2 = random.uniform(0.0, 1.0)
            edges.add((i, j))
            return PyTorchTableFactor.from_function(
                [x[i], x[j]],
                lambda xi, xj: (prob1 if xi else 1 - prob1) if xj else
                (prob2 if xi else 1 - prob2))

        def create_branch_start(i):
            depth[i] = 1
            prob1 = random.random()
            prob2 = random.random()
            return PyTorchTableFactor.from_function(
                [root, x[i]],
                lambda xi, xj: (prob1 if xi else 1 - prob1) if xj else
                (prob2 if xi else 1 - prob2))

        # create main branches up to minimum depth
        for i in range(0, branches):
            variable_index = i * min_depth
            model.append(create_branch_start(variable_index))
            model.extend([create_bivariate_factor(variable_index + j, variable_index + j + 1)
                          for j in range(0, min_depth - 1)])

        # attach remaining variables to random parents, excluding root
        for i in range(branches * min_depth, variable_count - 1):
            parent = random.randint(1, len(depth) - 1)
            model.append(create_bivariate_factor(parent, i))
            depth[i] = depth[parent] + 1

        # Add loop number of extra edges
        deep_nodes = [i for i in depth if depth[i] >= min_depth]
        for i in range(cross_edges):
            valid_selection = False
            while valid_selection:
                variables = random.sample(deep_nodes, k=2)
                ii = variables[0]
                jj = variables[1]
                valid_selection = not ((ii, jj) in edges or (jj, ii) in edges)
                if valid_selection:
                    model.append(create_bivariate_factor(ii, jj))
        return model, root

    (factors, query) = generate_deep_model(variable_count=81, branches=4, min_depth=20, cross_edges=40)

    evaluate_anytime(factors, query)


def evaluate_anytime(factors, query):
    exact_answer_time, exact_answer = measure_time(lambda: VariableElimination().run(query, factors))
    print()
    print(f"Exact answer: {exact_answer}")

    # We pick shallower nodes for expansion first since these are the most informative (breadth-first).
    def scoring_function(node, partial_tree, full_tree):
        return -full_tree.depth(node)

    anytime = AnytimeExactBeliefPropagation.from_factors(
        factors,
        query,
        expansion_value_function=scoring_function,
        approximation=message_approximation
    )
    true_query = {query: 1}

    def approximations_for_true_value(anytime, query):
        while not anytime.is_complete():
            yield anytime[query].normalize()(true_query).item()
            anytime.expand(query)

    exact_answer_for_true = exact_answer(true_query).item()
    full_ignorance_approximation = 0.5  # if we don't know anything we use the uniform distribution
    approximations = approximations_for_true_value(anytime, query)
    plot_approximations(f"Query {query}", exact_answer_for_true, exact_answer_time, full_ignorance_approximation,
                        approximations)


def plot_approximations(title, exact_answer, exact_answer_time, full_ignorance_approximation, approximations):
    anytime_relative_errors, times, anytime_absolute_errors = collect_anytime_data(approximations, exact_answer, exact_answer_time)
    absolute_error_in_full_ignorance = abs(full_ignorance_approximation - exact_answer)
    relative_error_in_full_ignorance = absolute_error_in_full_ignorance / exact_answer
    exact_method_absolute_errors = [absolute_error_in_full_ignorance if time < exact_answer_time else 0
                                    for time in times]
    exact_method_relative_errors = [relative_error_in_full_ignorance if time < exact_answer_time else 0
                                    for time in times]
    times_ms = [time * 1000 for time in times]

    # plt.subplot(1, 2, 1)
    # plt.title(f"{title}, exact answer = {exact_answer :.3}")
    # plt.xlabel("Time (ms)")
    # plt.ylabel("Relative error")
    # plt.plot(times_ms, anytime_relative_errors, 'r', label="Anytime")
    # plt.plot(times_ms, exact_method_relative_errors, 'b', label="Exact")
    # plt.legend(loc="upper right")
    #
    # plt.subplot(1, 2, 2)
    plt.title(f"{title}, exact answer = {exact_answer :.3}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Absolute error")
    plt.plot(times_ms, exact_method_absolute_errors, 'b', label="Exact")
    plt.plot(times_ms, anytime_absolute_errors, 'r', label="Anytime")
    plt.legend(loc="upper right")
    plt.show()


def collect_anytime_data(approximations, exact_answer, exact_answer_time):
    times = []
    absolute_errors = []
    relative_errors = []
    last_error = float("inf")
    starting_time = timeit.default_timer()
    for current_approximation in approximations:
        time = timeit.default_timer() - starting_time
        times.append(time)

        current_error = abs(current_approximation - exact_answer)
        absolute_errors.append(current_error)

        relative_error = current_error / exact_answer
        relative_errors.append(relative_error)

        improvement = last_error - current_error
        print(
            f"Approximation: {current_approximation}, current error: {current_error:.4f}, "
            f"last error: {last_error:.4f}, improvement: {improvement}")
        current_error_is_near_zero = current_error < 1e-3
        if current_error_is_near_zero and time > exact_answer_time:
            break
        else:
            last_error = current_error
    return relative_errors, times, absolute_errors
