from neuralpp.inference.graphical_model.learn.graphical_model_sgd_learner import GraphicalModelSGDLearner
from neuralpp.util.util import set_default_tensor_type_and_return_device, join


class LearningProblem:
    def make_data_loader(self):
        self._not_implemented()

    def setup_model(self):
        self._not_implemented()

    def print_evaluation(self):
        self._not_implemented()


def solve_learning_problem(
        problem: LearningProblem,
        try_cuda,
        lr,
        loss_decrease_tol,
        max_epochs_to_go_before_stopping_due_to_loss_decrease
):
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
