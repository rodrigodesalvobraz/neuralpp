from neuralpp.inference.graphical_model.representation.frame.dict_frame import (
    generalized_len_of_dict_frame,
    number_of_equal_values_in_dict_frames,
    to,
)
from neuralpp.inference.graphical_model.variable_elimination import VariableElimination
from neuralpp.util.util import mean


def cross_entropy_for_dataset(dataset, model, debug=False):
    return mean(
        cross_entropy_for_datapoint(observation, query_assignment, model, debug)
        for (observation, query_assignment) in dataset
    )


def cross_entropy_for_datapoint(observation_dict, query_dict, model, debug=False):
    probability = compute_query_probability(observation_dict, query_dict, model, debug)
    cross_entropy_loss = -probability.log().sum()
    return cross_entropy_loss


def compute_accuracy_on_frames_data_loader(data_loader, model, device):
    total_number_of_correct_predictions = 0
    total_number_of_predictions = 0
    for (observation_batch, query_batch) in data_loader:
        observation_batch = to(observation_batch, device)
        query_batch = to(query_batch, device)
        (
            number_of_correct_predictions,
            number_of_predictions,
        ) = compute_number_of_correct_and_total_predictions(
            observation_batch, query_batch, model
        )
        total_number_of_correct_predictions += number_of_correct_predictions
        total_number_of_predictions += number_of_predictions
    total_accuracy = total_number_of_correct_predictions / total_number_of_predictions
    return total_accuracy


def compute_number_of_correct_and_total_predictions(
    observation_dict, query_dict, model
):
    query_variables = query_dict.keys()
    prediction_dict = compute_query_prediction(observation_dict, query_variables, model)
    number_of_correct_predictions = number_of_equal_values_in_dict_frames(
        prediction_dict, query_dict
    )
    number_of_predictions = generalized_len_of_dict_frame(query_dict)
    return number_of_correct_predictions, number_of_predictions


def compute_query_prediction(observation_dict, query_variables, model):
    probability = compute_query_distribution(observation_dict, query_variables, model)
    prediction = probability.argmax()
    return prediction


def compute_query_probability(observation_dict, query_dict, model, debug=False):
    query_variables = query_dict.keys()
    query_distribution = compute_query_distribution(
        observation_dict, query_variables, model
    )
    probability = query_distribution[query_dict]

    if debug:
        # print(f"Observation: {observation_dict}")
        print(f"Target: {query_dict}")
        # print(f"Query distribution: {query_distribution}")
        print(f"Predicted probability of target: {probability}")

    return probability


def compute_query_distribution(observation_dict, query_variables, model):
    conditioned_model = [f.condition(observation_dict) for f in model]
    query_distribution = VariableElimination().run(query_variables, conditioned_model)
    return query_distribution
