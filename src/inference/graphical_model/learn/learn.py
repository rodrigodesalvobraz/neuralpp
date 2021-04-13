import time
from itertools import chain

import torch
from torch import autograd

from inference.graphical_model.representation.frame.dict_frame import generalized_len_of_dict_frames, to
from inference.graphical_model.representation.model.model import cross_entropy_for_datapoint


def default_learning_hook(**kwargs):
    # model, data_loader, device, previous_loss, loss, loss_decrease, loss_decrease_tol, time_start, end_str='\n'

    previous_loss = kwargs['previous_loss']
    time_start = kwargs['time_start']
    loss = kwargs['loss']
    end_str = kwargs.get('end_str', '\n')
    loss_decrease_tol = kwargs['loss_decrease_tol']

    include_decrease = previous_loss is not None
    time_elapsed = time.time() - time_start
    print(f"[{time_elapsed:.0f} s] Epoch loss: {loss:.4f}", end=' ' if include_decrease else end_str)
    if include_decrease:
        loss_decrease = previous_loss - loss
        print(f"(decrease: {loss_decrease:.6f}; stopping value is {loss_decrease_tol:.6f})", end=end_str)


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
              number_of_data_points_debug_limit=6,
              ):

        optimizer = torch.optim.Adam(chain(*(f.pytorch_parameters() for f in model)), lr=lr)
        loss = None
        previous_loss = None
        loss_decrease = None
        time_start = time.time()
        epoch = 0

        with torch.no_grad():
            if before_training is not None:
                before_training(
                    epoch=epoch,
                    model=model,
                    data_loader=data_loader,
                    device=device,
                    previous_loss=previous_loss,
                    loss=loss,
                    loss_decrease=loss_decrease,
                    loss_decrease_tol=loss_decrease_tol,
                    time_start=time_start)

        while not NeuralPPLearner.done(loss_decrease, loss_decrease_tol):
            total_loss = 0
            total_number_of_data_points = 0
            for (observation_batch, query_assignment_batch) in data_loader:
                if NeuralPPLearner.done(loss_decrease, loss_decrease_tol):
                    break
                observation_batch = to(observation_batch, device)
                query_assignment_batch = to(query_assignment_batch, device)
                number_of_data_points = generalized_len_of_dict_frames(observation_batch, query_assignment_batch)
                with autograd.set_detect_anomaly(False):
                    optimizer.zero_grad()
                    cross_entropy_loss = \
                        cross_entropy_for_datapoint(
                            observation_batch, query_assignment_batch, model,
                            debug=debug and total_number_of_data_points < number_of_data_points_debug_limit)
                    total_loss += cross_entropy_loss.item()
                    # backpropagate with retain_graph because components may be included more
                    # than once due to factors sharing parameters
                    cross_entropy_loss.backward(retain_graph=True)
                    optimizer.step()
                total_number_of_data_points += number_of_data_points

            loss = total_loss / total_number_of_data_points

            if previous_loss is not None:
                loss_decrease = previous_loss - loss

            with torch.no_grad():
                after_epoch(
                    epoch=epoch,
                    model=model,
                    data_loader=data_loader,
                    device=device,
                    previous_loss=previous_loss,
                    loss=loss,
                    loss_decrease=loss_decrease,
                    loss_decrease_tol=loss_decrease_tol,
                    time_start=time_start)

            previous_loss = loss
            epoch += 1
        return loss

    @staticmethod
    def done(loss_decrease, loss_decrease_tol):
        return loss_decrease is not None and loss_decrease <= loss_decrease_tol
