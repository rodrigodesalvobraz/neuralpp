from neuralpp.experiments.experimental_inference.exact_belief_propagation import ExactBeliefPropagation
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
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
    assert(ExactBeliefPropagation(conditioned_factors, w).run() == expected_w_with_conditions)


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
    assert(ExactBeliefPropagation(factors, w).run() == expected_w)

    # observe cloudiness at highest level
    observations = {c: 2}
    conditioned_factors = [f.condition(observations) for f in factors]

    # this should result in increased chances of rain
    expected_w_with_conditions = VariableElimination().run(w, conditioned_factors)
    assert(ExactBeliefPropagation(conditioned_factors, w).run() == expected_w_with_conditions)
