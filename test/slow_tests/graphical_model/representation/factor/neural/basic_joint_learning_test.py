import random

from inference.graphical_model.representation.model.model import cross_entropy_for_dataset, compute_query_probability
from inference.graphical_model.learn.learn import NeuralPPLearner
from inference.graphical_model.representation.factor.fixed.fixed_pytorch_factor import FixedPyTorchTableFactor
from inference.graphical_model.representation.factor.neural.MLPFactor import MLPFactor
from inference.graphical_model.representation.random.random_dataset import generate_dataset_given_observation_and_query_variables
from inference.graphical_model.variable.integer_variable import IntegerVariable
from util import util
from util.util import try_noisy_test_up_to_n_times


def joint_learning():

    x = IntegerVariable("x", 2)
    y = IntegerVariable("y", 2)
    z = IntegerVariable("z", 2)

    y_is_negation_of_x = FixedPyTorchTableFactor.from_function([x, y], lambda vx, vy: float(vy == 1 - vx))
    noise = 0.9
    z_noisy_equals_y = FixedPyTorchTableFactor.from_function([y, z], lambda vy, vz: noise if vz == vy else (1 - noise))

    ground_truth_model = [y_is_negation_of_x, z_noisy_equals_y]

    dataset = generate_dataset_given_observation_and_query_variables(
        ground_truth_model,
        observed_variables=[x],
        query_variables=[y, z],
        number_of_observations=100,
        datapoints_per_observation=1)

    print("Dataset sample:")
    print(util.join(random.sample(dataset, 20), "\n"))

    y_from_x = MLPFactor.make([x], y)
    learned_model = [y_from_x, z_noisy_equals_y]
    ground_truth_cross_entropy = cross_entropy_for_dataset(dataset, ground_truth_model).item()

    print(f"Ground truth model cross entropy: {ground_truth_cross_entropy :.3f}")

    NeuralPPLearner.learn(learned_model, dataset, loss_decrease_tol=1e-3)

    cross_entropy = cross_entropy_for_dataset(dataset, learned_model).item()

    print(f"Ground truth model cross entropy: {ground_truth_cross_entropy :.3f}")
    print(f"Learned model cross entropy     : {cross_entropy :.3f}")

    for observation_dict, query_dict in random.sample(dataset, 10):
        likelihood = compute_query_probability(observation_dict, query_dict, learned_model)
        print(f"Given {observation_dict}, answer is {query_dict}, learned model assigns it probability {likelihood.item():.2f}")

    expected_cross_entropy_upper_bound = ground_truth_cross_entropy * 1.2  # based on a few runs and observing that that's the usual approximation
    assert cross_entropy <= expected_cross_entropy_upper_bound, f"Learning performance was poorer than expected. Expected upper bound: {expected_cross_entropy_upper_bound:0.3f}, actual: {cross_entropy:.3f}"


def test_joint_learning():
    try_noisy_test_up_to_n_times(joint_learning, n=3)