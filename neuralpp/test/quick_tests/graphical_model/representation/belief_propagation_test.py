import random

from neuralpp.experiments.experimental_inference.approximations import (
    message_approximation,
)
from neuralpp.experiments.experimental_inference.exact_belief_propagation import (
    ExactBeliefPropagation,
    AnytimeExactBeliefPropagation,
)
from neuralpp.experiments.experimental_inference.graph_analysis import (
    FactorGraph,
)
from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
)
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.random.random_model import (
    generate_model,
)
from neuralpp.inference.graphical_model.variable.integer_variable import (
    IntegerVariable,
)
from neuralpp.inference.graphical_model.variable_elimination import (
    VariableElimination,
)

from matplotlib import pyplot as plt


def test_ebp_tree():
    prob_cloudy = [0.2, 0.4, 0.4]
    prob_sprinkler = [0.6, 0.4]
    prob_rain_given_cloudy = [
        [0.6, 0.3, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
    ]

    def prob_wet_grass(wetness: int, rain_level: int, sprinkler_on: int):
        return 1.0 if (rain_level + sprinkler_on == wetness) else 0.0

    c = IntegerVariable("c", 3)
    r = IntegerVariable("r", 3)
    s = IntegerVariable("s", 2)
    w = IntegerVariable("w", 4)

    factors = [
        PyTorchTableFactor([c], prob_cloudy),
        PyTorchTableFactor([c, r], prob_rain_given_cloudy),
        PyTorchTableFactor([s], prob_sprinkler),
        PyTorchTableFactor.from_function([w, r, s], prob_wet_grass),
    ]

    # expected_w = PyTorchTableFactor([w], [0.192, 0.332, 0.34, 0.136])
    expected_w = VariableElimination().run(w, factors)
    assert ExactBeliefPropagation(factors, w).run() == expected_w

    # observe cloudiness at highest level
    observations = {c: 2}
    conditioned_factors = [f.condition(observations) for f in factors]

    # this should result in increased chances of rain
    # expected_w_with_conditions = PyTorchTableFactor([w], [0.12, 0.26, 0.42, 0.2])
    expected_w_with_conditions = VariableElimination().run(w, conditioned_factors)
    assert (
        ExactBeliefPropagation(conditioned_factors, w).run()
        == expected_w_with_conditions
    )


def test_ebp_with_loop():
    prob_cloudy = [0.2, 0.4, 0.4]
    prob_rain_given_cloudy = [
        [0.6, 0.3, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
    ]
    prob_sprinkler_given_cloudy = [
        [0.2, 0.8],
        [0.5, 0.5],
        [0.8, 0.2],
    ]

    def prob_wet_grass(wetness: int, rain_level: int, sprinkler_on: int):
        return 1.0 if (rain_level + sprinkler_on == wetness) else 0.0

    c = IntegerVariable("c", 3)
    r = IntegerVariable("r", 3)
    s = IntegerVariable("s", 2)
    w = IntegerVariable("w", 4)

    factors = [
        PyTorchTableFactor([c], prob_cloudy),
        PyTorchTableFactor([c, r], prob_rain_given_cloudy),
        PyTorchTableFactor([c, s], prob_sprinkler_given_cloudy),
        PyTorchTableFactor.from_function([w, r, s], prob_wet_grass),
    ]

    expected_w = VariableElimination().run(w, factors)
    assert ExactBeliefPropagation(factors, w).run() == expected_w

    # observe cloudiness at highest level
    observations = {c: 2}
    conditioned_factors = [f.condition(observations) for f in factors]

    # this should result in increased chances of rain
    expected_w_with_conditions = VariableElimination().run(w, conditioned_factors)
    assert (
        ExactBeliefPropagation(conditioned_factors, w).run()
        == expected_w_with_conditions
    )


def test_random_model_ebp():
    for i in range(10):
        factors = generate_model(
            number_of_factors=15, number_of_variables=8, cardinality=3
        )
        query = random.choice([v for f in factors for v in f.variables])
        expected = VariableElimination().run(query, factors)
        assert ExactBeliefPropagation(factors, query).run() == expected


def test_incremental_anytime_with_uniform_approximation():
    prob_cloudy = [0.2, 0.4, 0.4]
    prob_sprinkler = [0.6, 0.4]
    prob_rain_given_cloudy = [
        [0.6, 0.3, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
    ]

    def prob_wet_grass(wetness: int, rain_level: int, sprinkler_on: int):
        return 1.0 if (rain_level + sprinkler_on == wetness) else 0.0

    c = IntegerVariable("c", 3)
    r = IntegerVariable("r", 3)
    s = IntegerVariable("s", 2)
    w = IntegerVariable("w", 4)

    factors = [
        PyTorchTableFactor([c], prob_cloudy),
        PyTorchTableFactor([c, r], prob_rain_given_cloudy),
        PyTorchTableFactor([s], prob_sprinkler),
        PyTorchTableFactor.from_function([w, r, s], prob_wet_grass),
    ]

    # Prefers nodes with more variable names later in the alphabet
    # Note: this requires variables to have one-character names.
    def scoring_function(x, partial_tree, full_tree):
        if isinstance(x, IntegerVariable):
            return ord(x.name)
        else:
            assert isinstance(x, Factor)
            return sum([ord(var.name) for var in x.variables])

    aebp_computation = AnytimeExactBeliefPropagation.from_factors(
        factors=factors,
        query=w,
        approximation=message_approximation,
        expansion_value_function=scoring_function,
    )

    approximations = []
    while not aebp_computation.is_complete():
        approximations.append(aebp_computation.run())
        aebp_computation.expand(w)
    approximations.append(aebp_computation.run())

    # First approximation ends up being uniform since all factors leading to the query are uniform
    # Final approximation is the same as the result from Exact Belief Propagation on the entire tree
    assert approximations[0] == PyTorchTableFactor([w], [0.25, 0.25, 0.25, 0.25])
    # assert (approximations[-1] == PyTorchTableFactor([w], [0.192, 0.332, 0.34, 0.136]))
    assert approximations[-1] == VariableElimination().run(w, factors)

    # Verify intermediates by constructing factor trees equivalent to the expected approximation step and
    # referencing with variable elimination
    variable_elimination_results = []

    def run_against_variable_elimination(approximate_factors):
        variable_elimination_results.append(
            VariableElimination().run(w, approximate_factors)
        )

    def uniform_on_variables_at(node):
        # TODO: Replace with a UniformFactor once this exists
        return PyTorchTableFactor.from_function(
            FactorGraph.variables_at(node), lambda *args: 1.0
        )

    uniform_r = uniform_on_variables_at(r)
    uniform_s = uniform_on_variables_at(s)
    uniform_c = uniform_on_variables_at(c)

    run_against_variable_elimination(
        [
            uniform_r,
            uniform_s,
            PyTorchTableFactor.from_function([w, r, s], prob_wet_grass),
        ]
    )

    run_against_variable_elimination(
        [
            uniform_r,
            PyTorchTableFactor([s], prob_sprinkler),
            PyTorchTableFactor.from_function([w, r, s], prob_wet_grass),
        ]
    )

    run_against_variable_elimination(
        [
            uniform_c,
            PyTorchTableFactor([c, r], prob_rain_given_cloudy),
            PyTorchTableFactor([s], prob_sprinkler),
            PyTorchTableFactor.from_function([w, r, s], prob_wet_grass),
        ]
    )

    # Some approximations share the same value, since approximations on a variable or a factor on that variable
    # create the same messages.
    assert approximations[1] == variable_elimination_results[0]
    assert approximations[2] == variable_elimination_results[0]
    assert approximations[3] == variable_elimination_results[1]
    assert approximations[4] == variable_elimination_results[1]
    assert approximations[5] == variable_elimination_results[2]
    assert approximations[6] == variable_elimination_results[2]


def test_random_model_aebp():
    # Test that complete runs of AEBP give the same result as variable elimination.
    # Note that this test does not verify any intermediate values.

    # Prefers nodes with more variable names later in the alphabet
    def scoring_function(x, partial_tree, full_tree):
        total_ord = lambda s: sum(ord(c) for c in s)  # noqa: E731
        if isinstance(x, IntegerVariable):
            return total_ord(x.name)
        else:
            assert isinstance(x, Factor)
            return sum([total_ord(var.name) for var in x.variables])

    def run_incremental_aebp_to_completion(factors, query):
        aebp = AnytimeExactBeliefPropagation.from_factors(
            factors,
            query,
            expansion_value_function=scoring_function,
            approximation=message_approximation,
        )
        while not aebp.is_complete():
            aebp.expand(query)
        return aebp.run()

    for i in range(50):
        factors = generate_model(
            number_of_factors=15, number_of_variables=8, cardinality=3
        )
        query = random.choice([v for f in factors for v in f.variables])
        expected = VariableElimination().run(query, factors)
        result = run_incremental_aebp_to_completion(factors, query)
        assert result == expected


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
    use_random_models = (
        False  # use a random model rather than the one described in the docstring.
    )
    number_of_factors = 25  # number of factors in the random model, if used.
    number_of_variables = int(
        number_of_factors * 2 / 3
    )  # number of variables in the random model, if used.

    # End of configuration

    # We pick shallower nodes for expansion first since these are the most informative (breadth-first).
    def scoring_function(node, partial_tree, full_tree):
        return -full_tree.depth(node)

    query = None

    if not use_random_models:
        # Linear models
        monotonic = True
        name = f"Linear model with {n-1} factors and {n} variables"
        x = [IntegerVariable(f"x{i}", 2) for i in range(n)]
        if not random_query:
            query = x[0]
        factors = [
            PyTorchTableFactor.from_function(
                [x[i], x[i + 1]],
                lambda xi, xip1: (0.9 if xi else 0.1) if xip1 else (0.2 if xi else 0.8),
            )
            for i in range(n - 1)
        ]
    else:
        # Random models
        name = f"Random model with {number_of_factors} factors and {number_of_variables} variables"
        monotonic = False
        factors = generate_model(
            number_of_factors=number_of_factors,
            number_of_variables=number_of_variables,
            cardinality=2,
        )

    if query is None:
        query = random.choice([v for f in factors for v in f.variables])

    anytime = AnytimeExactBeliefPropagation.from_factors(
        factors,
        query,
        expansion_value_function=scoring_function,
        approximation=message_approximation,
    )

    true_answer = VariableElimination().run(query, factors)

    print()
    print(f"True answer  : {true_answer}")

    last_error = float("inf")
    errors = []
    while True:
        current_approximation = anytime[query].normalize()
        true_query = {query: 1}
        current_error = abs(
            current_approximation(true_query) - true_answer(true_query)
        ).item()
        errors.append(current_error)
        improvement = last_error - current_error
        print(
            f"Approximation: {current_approximation}, current error: {current_error:.4f}, "
            f"last error: {last_error:.4f}, improvement: {improvement}"
        )
        if monotonic:
            assert improvement >= -1e-7

        if anytime.is_complete():
            break
        else:
            last_error = current_error
            anytime.expand(query)

    if show_plot:
        plt.title(f"{name}, query {query}")
        plt.plot(errors)
        plt.show()
