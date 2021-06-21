import torch

from torch import sigmoid

from neuralpp.inference.graphical_model.representation.factor.neural.neural_factor import NeuralFactor
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import PyTorchTableFactor
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable


class XorNeuralNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # sigmoid(first layer) represents 'or' and 'and' of inputs respectively.
        self.layer1 = torch.nn.Linear(2, 2)
        self.layer1.weight.data = torch.tensor([[10., 10.], [10., 10.]])
        self.layer1.bias.data = torch.tensor([-5., -15.])

        # sigmoid(second layer) represents values for xor = <or(inputs) and not(and(inputs))>
        self.layer2 = torch.nn.Linear(2, 1)
        self.layer2.weight.data = torch.tensor([[-10., 15.], [10., -15.]])
        self.layer2.bias.data = torch.tensor([5., -5.])

    def forward(self, x):
        if not torch.is_floating_point(x):
            x = x.float()
        x = self.layer1(x)
        x = sigmoid(x)
        x = self.layer2(x)
        x = sigmoid(x)
        return x

    def randomize(self):
        # TODO
        pass

    def randomized_copy(self):
        return self

    def __repr__(self):
        return "xor neural net"


def get_data():
    p = IntegerVariable("p", 2)
    q = IntegerVariable("q", 2)
    xor = IntegerVariable("xor", 2)
    neural_factor = NeuralFactor(XorNeuralNet(), [p, q], xor)
    return neural_factor, p, q, xor


def test_potentials():
    neural_factor, p, q, xor = get_data()
    print(neural_factor.table_factor)
    assert neural_factor.table_factor == \
           PyTorchTableFactor([p, q, xor], [[[0.9928, 0.0072], [0.0079, 0.9921]], [[0.0079, 0.9921], [0.9999, 0.0001]]])


def test_conditioning():
    neural_factor, p, q, xor = get_data()

    nothing = neural_factor.condition({})
    assert nothing == neural_factor

    not_p = neural_factor.condition({p: 0})
    print("xor | not p:", not_p)
    print("xor | not p table:", not_p.table_factor)
    assert not_p.table_factor == PyTorchTableFactor([q, xor], [[0.9928, 0.0072], [0.0079, 0.9921]])

    by_p = neural_factor.condition({p: 1})
    print("xor |     p:", by_p)
    print("xor |     p table:", by_p.table_factor)
    assert by_p.table_factor == PyTorchTableFactor([q, xor], [[0.0079, 0.9921], [0.9999, 0.0001]])

    p_and_q = neural_factor.condition({p: 1, q: 1})
    print("xor | p and q:", p_and_q)
    print("xor | p and q table:", p_and_q.table_factor)
    assert p_and_q.table_factor == PyTorchTableFactor([xor], [0.9999, 0.0001])

    xor_is_true = neural_factor.condition({xor: 1})
    print("xor | xor:", xor_is_true)
    print("xor | xor table:", xor_is_true.table_factor)
    assert xor_is_true.table_factor == PyTorchTableFactor([p, q], [[0.0072, 0.9921], [0.9921, 0.0]])

    xor_is_false = neural_factor.condition({xor: 0})
    print("xor | not xor:", xor_is_false)
    print("xor | not xor table:", xor_is_false.table_factor)
    assert xor_is_false.table_factor == PyTorchTableFactor([p, q], [[0.9928, 0.008], [0.008, 0.999]])

    p_and_xor_are_true = neural_factor.condition({p: 1, xor: 1})
    print("xor | p and xor:", p_and_xor_are_true)
    print("xor | p and xor table:", p_and_xor_are_true.table_factor)
    assert p_and_xor_are_true.table_factor == PyTorchTableFactor([q], [0.9921, 0.0])

    all_true = neural_factor.condition({p: 1, q: 1, xor: 1})
    print("xor | all true:", all_true)
    print("xor | all true table:", all_true.table_factor)
    assert all_true.table_factor == PyTorchTableFactor([], 0.0)


def test_normalize():
    neural_factor, p, q, xor = get_data()
    print("neural_factor.normalize()", neural_factor.normalize())
    assert neural_factor.normalize() == \
           PyTorchTableFactor([p, q, xor], [[[0.2482, 0.0018], [0.002, 0.248]], [[0.002, 0.248], [0.25, 0.00001]]])
