import random

from neuralpp.experiments.experimental_inference.approximations import uniform_approximation_fn
from neuralpp.experiments.experimental_inference.exact_belief_propagation import ExactBeliefPropagation, \
    IncrementalAnytimeBeliefPropagation
from neuralpp.inference.graphical_model.representation.factor.factor import Factor
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.random.random_model import generate_model
from neuralpp.inference.graphical_model.variable_elimination import VariableElimination
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable


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

    expected_w = PyTorchTableFactor([w], [0.192, 0.332, 0.34, 0.136])
    assert (ExactBeliefPropagation(factors, w).run() == expected_w)

    # observe cloudiness at highest level
    observations = {c: 2}
    conditioned_factors = [f.condition(observations) for f in factors]

    # this should result in increased chances of rain
    expected_w_with_conditions = PyTorchTableFactor([w], [0.12, 0.26, 0.42, 0.2])
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
    def scoring_function(x, partial_tree, full_tree):
        if isinstance(x, IntegerVariable):
            return ord(x.name)
        else:
            assert (isinstance(x, Factor))
            return sum([ord(var.name) for var in x.variables])

    aebp_computation = IncrementalAnytimeBeliefPropagation.from_factors(
        factors=factors,
        query=w,
        approximation_fn=uniform_approximation_fn,
        expansion_fn=scoring_function
    )

    approximations = []
    while not aebp_computation.is_complete():
        approximations.append(aebp_computation.run())
        aebp_computation.expand_partial_tree_and_recompute(w)
    approximations.append(aebp_computation.run())

    # First approximation ends up being uniform since all factors leading to the query are uniform
    # Final approximation is the same as the result from Exact Belief Propagation on the entire tree
    assert (approximations[0] == PyTorchTableFactor([w], [0.25, 0.25, 0.25, 0.25]))
    assert (approximations[-1] == PyTorchTableFactor([w], [0.192, 0.332, 0.34, 0.136]))


def test_random_model_aebp():
    # Prefers nodes with more variable names later in the alphabet
    def scoring_function(x, partial_tree, full_tree):
        total_ord = lambda s: sum(ord(c) for c in s)
        if isinstance(x, IntegerVariable):
            return total_ord(x.name)
        else:
            assert (isinstance(x, Factor))
            return sum([total_ord(var.name) for var in x.variables])

    def run_incremental_aebp_to_completion(factors, query):
        aebp = IncrementalAnytimeBeliefPropagation.from_factors(
            factors,
            query,
            expansion_fn=scoring_function,
            approximation_fn=uniform_approximation_fn
        )
        while not aebp.is_complete():
            aebp.expand_partial_tree_and_recompute(query)
        return aebp.run()

    for i in range(50):
        factors = generate_model(
            number_of_factors=15, number_of_variables=8, cardinality=3
        )
        query = random.choice([v for f in factors for v in f.variables])
        expected = VariableElimination().run(query, factors)
        result = run_incremental_aebp_to_completion(factors, query)
        assert (result == expected)
