from itertools import chain

from neuralpp.inference.graphical_model.representation.frame.dict_frame import generalized_len_of_dict_frames, to
from neuralpp.inference.graphical_model.representation.model.model import cross_entropy_for_datapoint
from neuralpp.util.generic_sgd_learner import GenericSGDLearner, default_after_epoch


class GraphicalModelSGDLearner(GenericSGDLearner):

    def __init__(self,
                 model,
                 data_loader,
                 lr=1e-3,
                 loss_decrease_tol=1e-3,
                 after_epoch=default_after_epoch,
                 debug=False,
                 device=None,
                 number_of_batches_between_updates=1,
                 max_epochs_to_go_before_stopping_due_to_loss_decrease=1):

        super().__init__(model,
                         data_loader,
                         lr,
                         loss_decrease_tol,
                         device,
                         number_of_batches_between_updates,
                         debug,
                         after_epoch,
                         max_epochs_to_go_before_stopping_due_to_loss_decrease=
                         max_epochs_to_go_before_stopping_due_to_loss_decrease)

    def batch_to_device(self, batch, device):
        (observation_batch, query_assignment_batch) = batch
        observation_batch = to(observation_batch, device)
        query_assignment_batch = to(query_assignment_batch, device)
        return (observation_batch, query_assignment_batch)

    def get_number_of_datapoints_in_batch(self, batch):
        (observation_batch, query_assignment_batch) = batch
        return generalized_len_of_dict_frames(observation_batch, query_assignment_batch)

    def loss_function(self, batch):
        (observation_batch, query_assignment_batch) = batch
        return cross_entropy_for_datapoint(observation_batch, query_assignment_batch, self.model, self.debug)

    def get_parameters(self, model):
        return chain(*(f.pytorch_parameters() for f in model))

    def must_retain_graph_during_backpropagation(self):
        """
        Backpropagate with retain_graph because components may be included more
        than once due to factors sharing parameters.
        """
        return True
