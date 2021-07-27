# import torch
# from torch.optim import Adam

# from neuralpp.inference.graphical_model.learn.graphical_model_sgd_learner import GraphicalModelSGDLearner
# from neuralpp.inference.graphical_model.variable.discrete_variable import DiscreteVariable
# from neuralpp.inference.graphical_model.variable.tensor_variable import TensorVariable

# from neuralpp.inference.neural_net.ConvNet import ConvNet
# from neuralpp.inference.neural_net.from_log_to_probabilities_adapter import FromLogToProbabilitiesAdapter
# from neuralpp.util.data_loader_from_random_data_point_thunk import data_loader_from_random_data_point_generator
# from neuralpp.util.mnist_util import read_mnist
# from neuralpp.util.util import set_default_tensor_type_and_return_device


# # This script tests whether it is possible for a ConvNet network to learn the digits 0 and 9
# # by negative feedback, namely telling them what they are *not*.
# # More specifically, when the input is the image of a digit other than 0, the epoch_average_loss is
# # the *negative* of the epoch_average_loss we would use to *predict* 0 (analogously for 9).
# # We expect the network to learn what is *not* a 0, and indirectly learn what *is*
# # (analogously for 9).
# # This script is a simplified version of mnist_semi_supervised.py.


# number_of_digits = 10  # actual script will use digits 0..number_of_digits and treat (number_of_digits - 1) as the '9'
#                        # in the experiment's description.
# number_of_batches_per_epoch = 100
# use_a_single_image_per_digit = False
# max_real_mnist_datapoints = None
# try_cuda = True


# def main():

#     neural_net = FromLogToProbabilitiesAdapter(ConvNet())
#     data_loader = data_loader_from_random_data_point_generator(number_of_batches_per_epoch, batch_generator, print=None)

#     global images_by_digits_by_phase  # so they are easily accessible in later functions
#     global next_image_index_by_digit
#     images_by_digits_by_phase = read_mnist(max_real_mnist_datapoints)
#     number_of_training_images = sum([len(images_by_digits_by_phase["train"][d]) for d in range(number_of_digits)])
#     print(f"Loaded {number_of_training_images:,} training images")
#     next_image_index_by_digit = {d: 0 for d in range(number_of_digits)}
#     from_digit_batch_to_image_batch = get_next_real_image_batch_for_digit_batch

#     device = set_default_tensor_type_and_return_device(try_cuda)
#     print(f"Using {device} device")

#     print("\nInitial evaluation:")
#     print_posterior_of(neural_net)

#     print("Learning...")

#     print("\nFinal evaluation:")
#     print_posterior_of(neural_net)


# def batch_generator():
#     pass


# def get_next_real_image_batch_for_digit_batch(digit_batch):
#     images_list = []
#     for d in digit_batch:
#         d = d.item()
#         image = images_by_digits_by_phase["train"][d][next_image_index_by_digit[d]]
#         if use_a_single_image_per_digit:
#             pass  # leave the index at the first position forever
#         else:
#             next_image_index_by_digit[d] += 1
#             if next_image_index_by_digit[d] == len(images_by_digits_by_phase["train"][d]):
#                 next_image_index_by_digit[d] = 0
#         images_list.append(image)
#     images_batch = torch.stack(images_list).to(digit_batch.device)
#     return images_batch


# def print_posterior_of(recognizer, **kwargs):
#     device = kwargs.get('device')
#     for digit in range(number_of_digits):
#         digit_batch = torch.tensor([digit])
#         image_batch = from_digit_batch_to_image_batch(digit_batch)
#         if device is not None:
#             digit_batch = digit_batch.to(device)
#         posterior_probability = recognizer(image_batch)
#         print_posterior(digit, posterior_probability)


# def print_posterior(digit, output_probability):
#     print(f"Prediction for image of {digit}: {output_probability}")
