import random

from inference.graphical_model.learn.learn import NeuralPPLearner
from inference.graphical_model.representation.frame.multi_frame_data_loader import MultiFrameDataLoader
from inference.graphical_model.representation.factor.fixed.fixed_pytorch_factor import FixedPyTorchTableFactor
from inference.graphical_model.representation.factor.neural.MLPFactor import MLPFactor
from inference.graphical_model.representation.model.model import cross_entropy_for_dataset
from inference.graphical_model.representation.random.random_dataset import generate_dataset
from inference.graphical_model.representation.random.random_model import generate_model
from util import util


def run_tests(
        number_of_tests,
        number_of_factors,
        number_of_variables,
        cardinality,
        fraction_of_replaced_factors,
        number_of_sets_of_observed_and_query_variables,
        number_of_query_variables,
        number_of_observed_variables,
        number_of_observations_per_random_set_of_observed_and_query_variables,
        datapoints_per_observation,
        loss_decrease_tol=1e-3,
        cross_entropy_ratio_tol=1.2,
        required_percentage = 0.75,
):

    test_results = [
        one_test(f"Starting test {i}",
                 number_of_factors,
                 number_of_variables,
                 cardinality,
                 fraction_of_replaced_factors,
                 number_of_sets_of_observed_and_query_variables,
                 number_of_query_variables,
                 number_of_observed_variables,
                 number_of_observations_per_random_set_of_observed_and_query_variables,
                 datapoints_per_observation,
                 loss_decrease_tol,
                 cross_entropy_ratio_tol)
        for i in range(number_of_tests)
    ]

    number_of_passing_tests = sum(test_results)

    print(f"Successful tests: {number_of_passing_tests} out of {number_of_tests}")

    assert number_of_passing_tests > required_percentage * number_of_tests, \
    f"Only {number_of_passing_tests} passed among {number_of_tests} tests. " \
    f"(required percentage: {required_percentage*100:.2}%). " \
    f"This might have been either a fluke or a real problem with the code."


def one_test(message,
             number_of_factors,
             number_of_variables,
             cardinality,
             fraction_of_replaced_factors,
             number_of_sets_of_observed_and_query_variables,
             number_of_query_variables,
             number_of_observed_variables,
             number_of_observations_per_random_set_of_observed_and_query_variables,
             datapoints_per_observation,
             loss_decrease_tol=1e-3,
             cross_entropy_ratio_tol=1.2):

    assert number_of_query_variables + number_of_observed_variables <= number_of_variables, f"number of query variables + number of observed variables must be less than number of variables, but got {number_of_query_variables}, {number_of_observed_variables}, {number_of_variables} respectively"

    print(message)

    ground_truth_model, learned_model = None, None

    while learned_model is None:
        ground_truth_model, learned_model = make_models(
            number_of_factors,
            number_of_variables,
            cardinality,
            fraction_of_replaced_factors)

        debug_model("Ground truth model:", ground_truth_model)
        debug_model("Initial state of model to be learned:", learned_model)

        number_of_replaced_factors = sum([l_f is not g_f for l_f, g_f in zip(learned_model, ground_truth_model)])

        print("Number of ground truth factors:", len(ground_truth_model))
        print("Number of replaced factors:", number_of_replaced_factors)

        if number_of_replaced_factors == 0:
            learned_model = None

    dataset = generate_dataset(
        ground_truth_model,
        number_of_sets_of_observed_and_query_variables,
        number_of_query_variables,
        number_of_observed_variables,
        number_of_observations_per_random_set_of_observed_and_query_variables,
        datapoints_per_observation)

    print(f"Generated random dataset from ground truth model of size {len(dataset)}")

    ground_truth_cross_entropy = cross_entropy_for_dataset(dataset, ground_truth_model).item()

    print(f"Ground truth model cross entropy: {ground_truth_cross_entropy :.3f}")

    data_loader = MultiFrameDataLoader(dataset, batch_size=10)
    cross_entropy = NeuralPPLearner.learn(learned_model, data_loader, loss_decrease_tol=loss_decrease_tol)

    print(f"Ground truth model cross entropy: {ground_truth_cross_entropy:.3f}")
    print(f"Learned model cross entropy     : {cross_entropy:.3f}")

    debug_model("Learned model after learning is completed:", learned_model)
    # print(join([(type(f).__name__, f) for f in learned_model], "\n"))

    expected_cross_entropy_upper_bound = ground_truth_cross_entropy * cross_entropy_ratio_tol
    if cross_entropy <= expected_cross_entropy_upper_bound:
        print("Test successful!")
        return True
    else:
        print(f"Learning performance was poorer than expected. "
              f"Expected upper bound: {expected_cross_entropy_upper_bound:0.3f}, "
              f"actual: {cross_entropy:.3f}")
        return False


def debug_model(message, model):
    pass
    # print(message)
    # for f in model:
    #     print(type(f).__name__)
    #     print(f)
    #     print("Parameters:", list(f.pytorch_parameters()))


def make_models(number_of_factors, number_of_variables, cardinality, fraction_of_replaced_factors):
    ground_truth_model = generate_model(number_of_factors, number_of_variables, cardinality, FixedPyTorchTableFactor)
    selected_for_replacement = set(util.select_indices_fraction(number_of_factors, fraction_of_replaced_factors))
    learned_model = [replace(ground_truth_model[i]) if i in selected_for_replacement else ground_truth_model[i]
                     for i in range(number_of_factors)]
    return ground_truth_model, learned_model


def replace(factor):
    if len(factor.variables) > 1:
        return make_neural_factor(factor).randomized_copy()
    else:
        return factor


def make_neural_factor(factor):
    assert len(factor.variables) > 0, f"To replace a factor by a neural factor, the original must have at least one variable, but got a factor without none: {factor}"
    output_variable = random.choice(factor.variables)
    input_variables = [v for v in factor.variables if v != output_variable]
    number_of_hidden_units = len(input_variables)
    neural_factor = MLPFactor.make(input_variables, number_of_hidden_units, output_variable)
    return neural_factor


def test_random_joint_learning():
    run_tests(
        number_of_tests=10,
        number_of_factors=8,
        number_of_variables=8,
        cardinality=3,
        fraction_of_replaced_factors=.5,
        number_of_sets_of_observed_and_query_variables=10,
        number_of_query_variables=1,
        number_of_observed_variables=2,
        number_of_observations_per_random_set_of_observed_and_query_variables=10,
        datapoints_per_observation=1,
        loss_decrease_tol=1e-2,
        cross_entropy_ratio_tol=1.2)
