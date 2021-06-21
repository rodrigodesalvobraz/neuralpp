class FromLogToProbabilitiesAdapter:

    def __init__(self, neural_net):
        self.neural_net = neural_net

    def to(self, device):
        return FromLogToProbabilitiesAdapter(self.neural_net.to(device))

    def __call__(self, input):
        logs = self.neural_net(input)
        return logs.exp()

    def parameters(self):
        return self.neural_net.parameters()

    def randomize(self):
        self.neural_net.randomize()

    def randomized_copy(self):
        return FromLogToProbabilitiesAdapter(self.neural_net.randomized_copy())

    def __repr__(self):
        return f"{self.neural_net}'s probabilities"
