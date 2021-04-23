import time

import torch
from torch import autograd
from tqdm import tqdm

from util.every_k_times import EveryKTimes


def default_after_epoch(learner, end_str='\n'):
    include_decrease = learner.previous_epoch_average_loss is not None
    time_elapsed = time.time() - learner.time_start
    end_line = ' ' if include_decrease else end_str
    print(f"[{time_elapsed:.0f} s] Epoch average loss: {learner.epoch_average_loss:.4f}", end=end_line)
    if include_decrease:
        loss_decrease = learner.previous_epoch_average_loss - learner.epoch_average_loss
        print(f"(decrease: {loss_decrease:.6f}; stopping value is {learner.loss_decrease_tol:.6f})", end=end_str)


class GenericSGDLearner:
    """
    Attributes available to all methods:
    model
    optimizer
    data_loader
    device
    detect_autograd_anomaly
    loss_decrease_tol
    lr
    number_of_batches_between_updates
    parameters
    retain_graph_during_backpropagation
    debug
    time_start
    epoch
    epoch_average_loss
    epoch_average_loss_decrease,
    total_epoch_loss
    total_number_of_data_points
    previous_epoch_average_loss
    loss_decrease_tol
    """

    def __init__(self, data_loader, device, loss_decrease_tol, lr, model, number_of_batches_between_updates,
                 debug=False, after_epoch=default_after_epoch):

        self.data_loader = data_loader
        self.device = device
        self.loss_decrease_tol = loss_decrease_tol
        self.lr = lr
        self.model = model
        self.number_of_batches_between_updates = number_of_batches_between_updates
        self.parameters = self.get_parameters(model)
        self.retain_graph_during_backpropagation = self.must_retain_graph_during_backpropagation()
        self.debug = debug
        self.after_epoch = after_epoch

        # Attributes to be initialized at learning time
        self.optimizer = None
        self.time_start = None
        self.epoch = None
        self.total_epoch_loss = None
        self.total_number_of_data_points = None
        self.epoch_average_loss = None
        self.previous_epoch_average_loss = None
        self.epoch_average_loss_decrease = None

    def batch_to_device(self, batch, device):
        """Default implementation assumes batch is a tensor and returns its copy on device"""
        return batch.to(device)

    def get_number_of_datapoints_in_batch(self, batch):
        return len(batch)

    def loss_function(self, batch):
        error = NotImplementedError(f"loss_function not implemented for {type(self)}")
        raise error

    def get_parameters(self, model):
        return model.parameters()

    def must_retain_graph_during_backpropagation(self):
        return False

    def learning_is_done(self):
        return self.epoch_average_loss_decrease is not None and self.epoch_average_loss_decrease <= self.loss_decrease_tol

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

        def update_parameters():
            self.optimizer.step()
            self.optimizer.zero_grad()

        update_parameters_every_k_batches = EveryKTimes(update_parameters, self.number_of_batches_between_updates)

        while not self.learning_is_done():
            print(f"Epoch: {self.epoch + 1}")
            self.total_epoch_loss = 0
            self.total_number_of_data_points = 0

            for batch_number, batch in tqdm(enumerate(self.data_loader, start=1)):
                batch = self.batch_to_device(batch, self.device)
                number_of_data_points = self.get_number_of_datapoints_in_batch(batch)
                batch_loss = self.loss_function(batch)
                batch_loss.backward(retain_graph=self.retain_graph_during_backpropagation)
                update_parameters_every_k_batches()
                self.total_epoch_loss += batch_loss.item()
                self.total_number_of_data_points += number_of_data_points

            self.epoch_average_loss = self.total_epoch_loss / self.total_number_of_data_points

            if self.previous_epoch_average_loss is not None:
                self.epoch_average_loss_decrease = self.previous_epoch_average_loss - self.epoch_average_loss

            with torch.no_grad():
                self.after_epoch(self)

            self.previous_epoch_average_loss = self.epoch_average_loss
            self.epoch += 1

        return self.epoch_average_loss
