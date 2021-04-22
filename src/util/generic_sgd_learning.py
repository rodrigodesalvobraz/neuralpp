import time

import torch
from torch import autograd
from tqdm import tqdm

from util.every_k_times import EveryKTimes


def generic_sgd_training(after_epoch, batch_to_device, before_training, data_loader, detect_autograd_anomaly, device,
                         get_number_of_datapoints_in_batch, learning_is_done, loss_decrease_tol, loss_function, lr, model,
                         number_of_batches_between_updates, parameters, retain_graph_during_backpropagation):
    optimizer = torch.optim.Adam(parameters, lr=lr)
    optimizer.zero_grad()
    loss = None
    previous_loss = None
    loss_decrease = None
    time_start = time.time()
    epoch = 0

    def update_parameters():
        optimizer.step()
        optimizer.zero_grad()

    update_parameters_every_k_batches = EveryKTimes(update_parameters, number_of_batches_between_updates)
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
    with autograd.set_detect_anomaly(detect_autograd_anomaly):
        while not learning_is_done(loss_decrease):
            print(f"Epoch: {epoch + 1}")
            total_loss = 0
            total_number_of_data_points = 0
            for batch_number, batch in tqdm(enumerate(data_loader, start=1)):
                batch = batch_to_device(batch, device)
                number_of_data_points = get_number_of_datapoints_in_batch(batch)
                loss = loss_function(batch)
                total_loss += loss.item()
                loss.backward(retain_graph=retain_graph_during_backpropagation)
                update_parameters_every_k_batches()
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