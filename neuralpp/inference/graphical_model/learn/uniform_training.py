import torch

from neuralpp.util.generic_sgd_learner import GenericSGDLearner


class UniformTraining(GenericSGDLearner):
    def __init__(self, model, data_loader, number_of_classes):
        super().__init__(model, data_loader)
        self.number_of_classes = number_of_classes
        self.singleton_probability = [1.0 / self.number_of_classes]

    def loss_function(self, batch):
        probabilities = self.model(batch)
        uniform_probabilities = torch.tensor(self.singleton_probability).repeat(
            len(batch), self.number_of_classes
        )
        loss = torch.square(uniform_probabilities - probabilities).sum()
        return loss
