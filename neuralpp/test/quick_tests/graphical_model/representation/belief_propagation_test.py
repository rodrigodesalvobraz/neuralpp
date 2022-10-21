import random

from neuralpp.experiments.experimental_inference.approximations import message_approximation
from neuralpp.experiments.experimental_inference.exact_belief_propagation import ExactBeliefPropagation, \
    AnytimeExactBeliefPropagation
from neuralpp.experiments.experimental_inference.graph_analysis import FactorGraph
from neuralpp.inference.graphical_model.representation.factor.factor import Factor
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.random.random_model import generate_model
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.graphical_model.variable_elimination import VariableElimination


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
    assert (ExactBeliefPropagation(factors, w).run() == expected_w)

    # observe cloudiness at highest level
    observations = {c: 2}
    conditioned_factors = [f.condition(observations) for f in factors]

    # this should result in increased chances of rain
    # expected_w_with_conditions = PyTorchTableFactor([w], [0.12, 0.26, 0.42, 0.2])
    expected_w_with_conditions = VariableElimination().run(w, conditioned_factors)
    assert (ExactBeliefPropagation(conditioned_factors, w).run() == expected_w_with_conditions)


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
    assert (ExactBeliefPropagation(factors, w).run() == expected_w)

    # observe cloudiness at highest level
    observations = {c: 2}
    conditioned_factors = [f.condition(observations) for f in factors]

    # this should result in increased chances of rain
    expected_w_with_conditions = VariableElimination().run(w, conditioned_factors)
    assert (ExactBeliefPropagation(conditioned_factors, w).run() == expected_w_with_conditions)


def test_random_model_ebp():
    for i in range(10):
        factors = generate_model(
            number_of_factors=15, number_of_variables=8, cardinality=3
        )
        query = random.choice([v for f in factors for v in f.variables])
        expected = VariableElimination().run(query, factors)
        assert (ExactBeliefPropagation(factors, query).run() == expected)


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
            assert (isinstance(x, Factor))
            return sum([ord(var.name) for var in x.variables])

    aebp_computation = AnytimeExactBeliefPropagation.from_factors(
        factors=factors,
        query=w,
        approximation=message_approximation,
        expansion_value_function=scoring_function
    )

    approximations = []
    while not aebp_computation.is_complete():
        approximations.append(aebp_computation.run())
        aebp_computation.expand(w)
    approximations.append(aebp_computation.run())

    # First approximation ends up being uniform since all factors leading to the query are uniform
    # Final approximation is the same as the result from Exact Belief Propagation on the entire tree
    assert (approximations[0] == PyTorchTableFactor([w], [0.25, 0.25, 0.25, 0.25]))
    # assert (approximations[-1] == PyTorchTableFactor([w], [0.192, 0.332, 0.34, 0.136]))
    assert (approximations[-1] == VariableElimination().run(w, factors))

    # Verify intermediates by constructing factor trees equivalent to the expected approximation step and
    # referencing with variable elimination
    variable_elimination_results = []

    def run_against_variable_elimination(approximate_factors):
        variable_elimination_results.append(VariableElimination().run(w, approximate_factors))

    def uniform_on_variables_at(node):
        # TODO: Replace with a UniformFactor once this exists
        return PyTorchTableFactor.from_function(FactorGraph.variables_at(node), lambda *args: 1.0)

    uniform_r = uniform_on_variables_at(r)
    uniform_s = uniform_on_variables_at(s)
    uniform_c = uniform_on_variables_at(c)

    run_against_variable_elimination([
        uniform_r,
        uniform_s,
        PyTorchTableFactor.from_function([w, r, s], prob_wet_grass)
    ])

    run_against_variable_elimination([
        uniform_r,
        PyTorchTableFactor([s], prob_sprinkler),
        PyTorchTableFactor.from_function([w, r, s], prob_wet_grass)
    ])

    run_against_variable_elimination([
        uniform_c,
        PyTorchTableFactor([c, r], prob_rain_given_cloudy),
        PyTorchTableFactor([s], prob_sprinkler),
        PyTorchTableFactor.from_function([w, r, s], prob_wet_grass)
    ])

    # Some approximations share the same value, since approximations on a variable or a factor on that variable
    # create the same messages.
    assert (approximations[1] == variable_elimination_results[0])
    assert (approximations[2] == variable_elimination_results[0])
    assert (approximations[3] == variable_elimination_results[1])
    assert (approximations[4] == variable_elimination_results[1])
    assert (approximations[5] == variable_elimination_results[2])
    assert (approximations[6] == variable_elimination_results[2])


def test_random_model_aebp():
    # Test that complete runs of AEBP give the same result as variable elimination.
    # Note that this test does not verify any intermediate values.

    # Prefers nodes with more variable names later in the alphabet
    def scoring_function(x, partial_tree, full_tree):
        total_ord = lambda s: sum(ord(c) for c in s)
        if isinstance(x, IntegerVariable):
            return total_ord(x.name)
        else:
            assert (isinstance(x, Factor))
            return sum([total_ord(var.name) for var in x.variables])

    def run_incremental_aebp_to_completion(factors, query):
        aebp = AnytimeExactBeliefPropagation.from_factors(
            factors,
            query,
            expansion_value_function=scoring_function,
            approximation=message_approximation
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
        assert (result == expected)
