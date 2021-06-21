import random

from neuralpp.inference.graphical_model.learn.graphical_model_sgd_learner import GraphicalModelSGDLearner
from neuralpp.inference.graphical_model.representation.factor.neural.neural_factor import NeuralFactor
from neuralpp.test.slow_tests.graphical_model.representation.factor.neural.neural_factor_test_util import check_and_show_conditional_distributions
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.neural_net.MLP import MLP
from neuralpp.util import util
from neuralpp.util.util import try_noisy_test_up_to_n_times


def make_bernoulli_dataset(x, y):
    global dataset

    def bernoulli(p):
        return int(random.random() < p)

    DATASET_SIZE = 1000
    p = [0.3, 0.9]

    x_value = lambda i: 0 if i < DATASET_SIZE/2 else 1
    y_value = lambda i: bernoulli(p[x_value(i)])
    dataset = [({x: x_value(i)}, {y: y_value(i)}) for i in range(DATASET_SIZE)]
    random.shuffle(dataset)
    return dataset


def mlp_neural_factor_learning():
    x = IntegerVariable("x", 2)
    y = IntegerVariable("y", 2)
    input_variables = [x]
    output_variable = y
    neural_net = MLP(len(input_variables), output_variable.cardinality)
    neural_net = neural_net.randomized_copy()
    neural_factor = NeuralFactor(neural_net, input_variables, output_variable)

    dataset = make_bernoulli_dataset(x, y)
    print()
    print("Dataset first few lines:")
    print(util.join(dataset[:10], "\n"))
    print()
    y_for_x_0 = [row[1][y] for row in dataset if row[0][x] == 0]
    y_for_x_1 = [row[1][y] for row in dataset if row[0][x] == 1]
    # print("y for x = 0:", y_for_x_0)
    # print("y for x = 1:", y_for_x_1)
    y_mean_for_x_0 = sum(y_for_x_0)/len(y_for_x_0)
    y_mean_for_x_1 = sum(y_for_x_1)/len(y_for_x_1)
    print("Dataset mean y for x = 0:", y_mean_for_x_0)
    print("Dataset mean y for x = 1:", y_mean_for_x_1)

    graphical_model = [neural_factor]
    GraphicalModelSGDLearner(graphical_model, dataset, loss_decrease_tol=1e-3).learn()

    def ground_truth(x, y):
        if x == 0:
            return y_mean_for_x_0 if y == 1 else 1 - y_mean_for_x_0
        else:
            return y_mean_for_x_1 if y == 1 else 1 - y_mean_for_x_1

    check_and_show_conditional_distributions(neural_factor, ground_truth)


def test_joint_learning():
    try_noisy_test_up_to_n_times(mlp_neural_factor_learning, n=3)
