import time

import torch
from neuralpp.util.every_k_times import EveryKTimes
from neuralpp.util.util import join
from torch import autograd
from tqdm import tqdm


# def default_after_epoch(learner, end_str='\n'):
#     include_decrease = learner.previous_epoch_average_loss is not None
#     time_elapsed = time.time() - learner.time_start
#     end_line = ' ' if include_decrease else end_str
#     print(f"[{time_elapsed:.0f} s] Epoch average loss: {learner.epoch_average_loss:.4f}", end=end_line)
#     if include_decrease:
#         loss_decrease = learner.previous_epoch_average_loss - learner.epoch_average_loss
#         print(f"(decrease: {loss_decrease:.6f}; stopping value is {learner.loss_decrease_tol:.6f})", end=end_str)


def default_after_epoch(learner, end_str="\n"):
    time_elapsed = time.time() - learner.time_start
    main_info = f"[{time_elapsed:.0f} s] Epoch average loss: {learner.epoch_average_loss:.4f}"
    extra_items = []
    include_decrease = learner.previous_epoch_average_loss is not None
    if include_decrease:
        loss_decrease = (
            learner.previous_epoch_average_loss - learner.epoch_average_loss
        )
        extra_items.append(f"decrease: {loss_decrease:.6f}")
        extra_items.append(
            f"stopping value is {learner.loss_decrease_tol:.6f}"
        )
    if learner.max_epochs_to_go_before_stopping_due_to_loss_decrease > 1:
        extra_items.append(
            f"number of epochs without less decrease: {learner.number_of_epochs_without_loss_decrease}"
        )
        extra_items.append(
            f"max number of epochs without less decrease: {learner.max_epochs_to_go_before_stopping_due_to_loss_decrease}"
        )
    final_message = (
        main_info + " (" + join(extra_items, "; ") + ")"
        if extra_items
        else main_info
    )
    print(final_message, end=end_str)


class GenericSGDLearner:
    """
    A generic implementation of Stochastic Gradient Descent that does not make many assumptions regarding the
    actual data and model being learned.

    Attributes available to all methods:
    model
    data_loader
    lr
    loss_decrease_tol
    number_of_batches_between_updates
    device
    debug
    max_epochs_to_go_before_stopping_due_to_loss_decrease

    optimizer
    detect_autograd_anomaly
    parameters
    retain_graph_during_backpropagation
    time_start
    epoch
    epoch_average_loss
    epoch_average_loss_decrease,
    total_epoch_loss
    total_number_of_data_points
    previous_epoch_average_loss
    """

    def __init__(
        self,
        model,
        data_loader,
        lr=1e-3,
        loss_decrease_tol=1e-3,
        device=None,
        number_of_batches_between_updates=1,
        debug=False,
        after_epoch=default_after_epoch,
        max_epochs_to_go_before_stopping_due_to_loss_decrease=1,
    ):

        self.model = model
        self.data_loader = data_loader
        self.lr = lr
        self.loss_decrease_tol = loss_decrease_tol
        self.device = device
        self.number_of_batches_between_updates = (
            number_of_batches_between_updates
        )
        self.debug = debug
        self.after_epoch = after_epoch
        self.max_epochs_to_go_before_stopping_due_to_loss_decrease = (
            max_epochs_to_go_before_stopping_due_to_loss_decrease
        )

        self.parameters = self.get_parameters(model)
        self.retain_graph_during_backpropagation = (
            self.must_retain_graph_during_backpropagation()
        )

        # Attributes to be initialized at learning time
        self.optimizer = None
        self.time_start = None
        self.epoch = None
        self.total_epoch_loss = None
        self.total_number_of_data_points = None
        self.epoch_average_loss = None
        self.previous_epoch_average_loss = None
        self.epoch_average_loss_decrease = None
        self.number_of_epochs_without_loss_decrease = None

    def batch_to_device(self, batch, device):
        """Default implementation assumes batch is a tensor and returns its copy on device"""
        return batch.to(device)

    def get_number_of_datapoints_in_batch(self, batch):
        return len(batch)

    def loss_function(self, batch):
        error = NotImplementedError(
            f"loss_function not implemented for {type(self)}"
        )
        raise error

    def get_parameters(self, model):
        return model.parameters()

    def must_retain_graph_during_backpropagation(self):
        return False

    def learning_is_done(self):
        no_loss_decrease = (
            self.epoch_average_loss_decrease is not None
            and self.epoch_average_loss_decrease <= self.loss_decrease_tol
        )
        if no_loss_decrease:
            self.number_of_epochs_without_loss_decrease += 1
        else:
            self.number_of_epochs_without_loss_decrease = 0
        done = (
            self.number_of_epochs_without_loss_decrease
            == self.max_epochs_to_go_before_stopping_due_to_loss_decrease
        )
        return done

    def learn(self):

        self.optimizer = torch.optim.Adam(self.parameters, lr=self.lr)
        self.optimizer.zero_grad()
        self.time_start = time.time()
        self.epoch = 0
        self.total_epoch_loss = None
        self.total_number_of_data_points = None
        self.epoch_average_loss = None
        self.previous_epoch_average_loss = None
        self.epoch_average_loss_decrease = None
        self.number_of_epochs_without_loss_decrease = 0

        def update_parameters():
            self.optimizer.step()
            self.optimizer.zero_grad()

        update_parameters_every_k_batches = EveryKTimes(
            update_parameters, self.number_of_batches_between_updates
        )

        while not self.learning_is_done():
            print(f"Epoch: {self.epoch + 1}")
            self.total_epoch_loss = 0
            self.total_number_of_data_points = 0

            for batch_number, batch in tqdm(
                enumerate(self.data_loader, start=1)
            ):
                batch = self.batch_to_device(batch, self.device)
                number_of_data_points = (
                    self.get_number_of_datapoints_in_batch(batch)
                )
                batch_loss = self.loss_function(batch)
                batch_loss.backward(
                    retain_graph=self.retain_graph_during_backpropagation
                )
                update_parameters_every_k_batches()
                self.total_epoch_loss += batch_loss.item()
                self.total_number_of_data_points += number_of_data_points

            self.epoch_average_loss = (
                self.total_epoch_loss / self.total_number_of_data_points
            )

            if self.previous_epoch_average_loss is not None:
                self.epoch_average_loss_decrease = (
                    self.previous_epoch_average_loss - self.epoch_average_loss
                )

            with torch.no_grad():
                self.after_epoch(self)

            self.previous_epoch_average_loss = self.epoch_average_loss
            self.epoch += 1

        return self.epoch_average_loss
