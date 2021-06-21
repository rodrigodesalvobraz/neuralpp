from neuralpp.inference.graphical_model.learn.graphical_model_sgd_learner import GraphicalModelSGDLearner
from neuralpp.inference.graphical_model.representation.model.model import cross_entropy_for_dataset
from neuralpp.inference.graphical_model.representation.random.random_dataset import generate_dataset
from neuralpp.inference.graphical_model.representation.random.random_model import generate_model
from neuralpp.util.util import join, assert_equal_up_to_relative_tolerance, try_noisy_test_up_to_n_times


def noisy_test():
    ground_truth_model = generate_model(number_of_factors=6, number_of_variables=6, cardinality=3)
    print("Model:")
    print(join(ground_truth_model, "\n"))
    print()

    ds = generate_dataset(ground_truth_model, number_of_sets_of_observed_and_query_variables=50,
                          number_of_query_variables=2, number_of_observed_variables=2,
                          number_of_observations_per_random_set_of_observed_and_query_variables=1,
                          datapoints_per_observation=10)
    print()
    print(f"Generated dataset (5 out of {len(ds)} first rows):")
    print(join(ds[:5], "\n"))

    cross_entropy_for_ground_model = cross_entropy_for_dataset(ds, ground_truth_model)
    print()
    print(f"Ground truth model loss (cross entropy) for this dataset: {cross_entropy_for_ground_model:.3f}")

    m = [f.randomized_copy() for f in ground_truth_model]

    device = None
    GraphicalModelSGDLearner(m, ds, loss_decrease_tol=1e-2, device=device).learn()

    for f1, f2 in zip(ground_truth_model, m):
        print()
        print(f"Original factor: {f1.normalize()}")
        print(f"Learned  factor: {f2.normalize()}")

    cross_entropy = cross_entropy_for_dataset(ds, m).item()

    print(f"\nGround truth  cross entropy: {cross_entropy_for_ground_model:.3f}")
    print(f"Learned model cross entropy: {cross_entropy:.3f}")

    assert_equal_up_to_relative_tolerance(cross_entropy, cross_entropy_for_ground_model, 0.1)


def test_learn():
    try_noisy_test_up_to_n_times(noisy_test, n=3)
