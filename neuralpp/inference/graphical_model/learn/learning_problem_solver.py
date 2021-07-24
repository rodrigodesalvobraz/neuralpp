from neuralpp.inference.graphical_model.learn.graphical_model_sgd_learner import GraphicalModelSGDLearner
from neuralpp.util.util import set_default_tensor_type_and_return_device, join

# This module provides a generic function for solving learning problems.

# It provides a LearningProblem interface for users to extend with the details of their
# problems.
# They can then invoke function solve_learning_problem on their learning problem
# and have it solved.
# See more details in the documentation of each element.

class LearningProblem:
    """
    An interface for learning problems.
    It must know how to provide a data loader, set up a model, evaluate the model,
    and print summaries after each learning epoch.
    """

    def make_data_loader(self):
        """
        Provides a data loader for the problem.
        The data loader must be an iterable over data points or batches
        that can be provided to the self.model as arguments.
        """
        raise NotImplementedError("make_data_loader")

    def setup_model(self):
        """Sets up self.model."""
        raise NotImplementedError("setup_model")

    def learning_is_needed(self):
        """
        Indicates whether learning is actually needed
        (must return False if self.model is already created
        in final form).
        """
        raise NotImplementedError("learning_is_needed")

    def print_evaluation(self):
        """Evaluates self.model, printing a summary."""
        raise NotImplementedError("print_evaluation")

    @property
    def after_epoch(self):
        """See documentation for GraphicalModelSGDLearner after_epoch parameter."""
        raise NotImplementedError("after_epoch")


def solve_learning_problem(
        problem: LearningProblem,
        try_cuda,
        lr,
        loss_decrease_tol,
        max_epochs_to_go_before_stopping_due_to_loss_decrease
):
    """
    Solves a LearningProblem.
    Runs an initial evaluation before learning starts, and another at the end.
    Uses GraphicalModelSGDLearner, passing it problem.model, data loader
    and after_epoch provided by the learning problem.
    """

    # Load data, if needed, before setting default device to cuda
    train_data_loader = problem.make_data_loader()

    device = set_default_tensor_type_and_return_device(try_cuda)

    # Creating model after attempting to set default tensor type to cuda so it sits there
    problem.setup_model()

    print("\nInitial evaluation:")
    problem.print_evaluation()

    if problem.learning_is_needed():
        print("Learning...")
        learner = GraphicalModelSGDLearner(
            problem.model,
            train_data_loader,
            device=device,
            lr=lr,
            loss_decrease_tol=loss_decrease_tol,
            max_epochs_to_go_before_stopping_due_to_loss_decrease=max_epochs_to_go_before_stopping_due_to_loss_decrease,
            after_epoch=problem.after_epoch,
        )
        learner.learn()

    print("\nFinal model:")
    print(join(problem.model, "\n"))

    print("\nFinal evaluation:")
    problem.print_evaluation()
