import random

from neuralpp.inference.graphical_model.representation.random.random_model import generate_model
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.graphical_model.variable_elimination import VariableElimination
from neuralpp.util.util import join, repeat


def generate_assignment_dict(variables):

    assert all(isinstance(v, IntegerVariable) for v in variables), \
        f"Random dataset generation is currently defined for {IntegerVariable.__name__} only"

    return {v: random.randrange(0, v.cardinality) for v in variables}


def condition(model, observation_dict):
    return [factor.condition(observation_dict) for factor in model]


def generate_dataset(model,
                     number_of_sets_of_observed_and_query_variables,
                     number_of_query_variables,
                     number_of_observed_variables,
                     number_of_observations_per_random_set_of_observed_and_query_variables,
                     datapoints_per_observation):
    model_variables = {v for factor in model for v in factor.variables}
    dataset = []
    for i in range(number_of_sets_of_observed_and_query_variables):
        observation_dataset = \
            generate_dataset_given_number_of_observation_and_query_variables(
                model,
                model_variables,
                number_of_query_variables,
                number_of_observed_variables,
                number_of_observations_per_random_set_of_observed_and_query_variables,
                datapoints_per_observation)
        dataset = dataset + observation_dataset
    return dataset


def generate_dataset_given_number_of_observation_and_query_variables(
        model,
        model_variables,
        number_of_query_variables,
        number_of_observed_variables,
        number_of_observations_per_random_set_of_observed_and_query_variables,
        datapoints_per_observation):

    observed_variables = set(random.sample(model_variables, number_of_observed_variables))
    query_variables = random.sample(list(model_variables - observed_variables), number_of_query_variables)

    return generate_dataset_given_observation_and_query_variables(
        model,
        observed_variables,
        query_variables,
        number_of_observations_per_random_set_of_observed_and_query_variables,
        datapoints_per_observation=datapoints_per_observation)


def generate_dataset_given_observation_and_query_variables(model, observed_variables, query_variables,
                                                           number_of_observations, datapoints_per_observation):
    dataset = []
    for i in range(number_of_observations):
        observation_dict = generate_assignment_dict(observed_variables)
        dataset_for_observation_dict = \
            generate_dataset_given_observation_dict_and_query_variables(datapoints_per_observation, model,
                                                                        observation_dict, query_variables)
        dataset.extend(dataset_for_observation_dict)
    return dataset

def generate_dataset_given_observation_dict_and_query_variables(datapoints_per_observation, model, observation_dict,
                                                                query_variables):
    conditioned_model = condition(model, observation_dict)
    query_distribution = VariableElimination().run(query_variables, conditioned_model).atomic_factor()
    observation_dataset = \
        repeat(datapoints_per_observation, lambda: (observation_dict, query_distribution.single_sample_assignment_dict()))
    # print(f"observation_dict: {observation_dict}")
    # print(f"conditioned_model: {conditioned_model}")
    # print(f"query_variables: {query_variables}")
    # print(f"query_distribution: {query_distribution}")
    # observation_dataset_string = join(observation_dataset, '\n')
    # print(f"observation_dataset:\n{observation_dataset_string}")
    return observation_dataset


if __name__ == '__main__':
    model = generate_model(number_of_factors=6, number_of_variables=4, cardinality=3)
    print("Model:")
    print(join(model, "\n"))
    print()

    ds = generate_dataset(model=model, number_of_sets_of_observed_and_query_variables=5,
                          number_of_query_variables=random.randint(0, 2), number_of_observed_variables=2,
                          number_of_observations_per_random_set_of_observed_and_query_variables=1,
                          datapoints_per_observation=10)
    print()
    print("Generated dataset:")
    print(join(ds, "\n"))
