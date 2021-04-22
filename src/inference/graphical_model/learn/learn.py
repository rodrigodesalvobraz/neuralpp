from itertools import chain

from inference.graphical_model.representation.frame.dict_frame import generalized_len_of_dict_frames, to
from inference.graphical_model.representation.model.model import cross_entropy_for_datapoint
from util.generic_sgd_learning import generic_sgd_training, default_learning_hook


class NeuralPPLearner:

    @staticmethod
    def learn(model,
              data_loader,
              lr=1e-3,
              loss_decrease_tol=1e-3,
              after_epoch=default_learning_hook,
              debug=False,
              device=None,
              before_training=None,
              number_of_batches_between_updates=1,
              ):

        def neupps_batch_to_device(batch, device):
            (observation_batch, query_assignment_batch) = batch
            observation_batch = to(observation_batch, device)
            query_assignment_batch = to(query_assignment_batch, device)
            return (observation_batch, query_assignment_batch)

        def get_number_of_datapoints_in_neupps_batch(batch):
            (observation_batch, query_assignment_batch) = batch
            return generalized_len_of_dict_frames(observation_batch, query_assignment_batch)

        def loss_function_of_neupps_batch(batch):
            (observation_batch, query_assignment_batch) = batch
            return cross_entropy_for_datapoint(observation_batch, query_assignment_batch, model, debug)

        neupps_parameters = chain(*(f.pytorch_parameters() for f in model))

        def neupps_learning_is_done(loss_decrease):
            return loss_decrease is not None and loss_decrease <= loss_decrease_tol

        # backpropagate with retain_graph because components may be included more
        # than once due to factors sharing parameters
        neupps_must_retain_graph_during_backpropagation = True

        detect_autograd_anomaly_for_neupps = False

        # End of Neupps-specific settings

        retain_graph_during_backpropagation = neupps_must_retain_graph_during_backpropagation
        get_number_of_datapoints_in_batch = get_number_of_datapoints_in_neupps_batch
        batch_to_device = neupps_batch_to_device
        loss_function = loss_function_of_neupps_batch
        detect_autograd_anomaly = detect_autograd_anomaly_for_neupps
        learning_is_done = neupps_learning_is_done
        parameters = neupps_parameters

        loss = generic_sgd_training(after_epoch, batch_to_device, before_training, data_loader,
                                    detect_autograd_anomaly, device, get_number_of_datapoints_in_batch,
                                    learning_is_done, loss_decrease_tol, loss_function, lr, model,
                                    number_of_batches_between_updates, parameters,
                                    retain_graph_during_backpropagation)
        return loss
