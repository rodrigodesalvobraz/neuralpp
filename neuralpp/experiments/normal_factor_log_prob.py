import torch
import torch.distributions as dist

from neuralpp.inference.graphical_model.representation.factor.continuous.normal_factor import (
    NormalFactor,
)
from neuralpp.inference.graphical_model.variable.tensor_variable import (
    TensorVariable,
)

value = TensorVariable("value", 0)
loc = TensorVariable("loc", 0)
scale = TensorVariable("scale", 0)

normal_factor = NormalFactor([value, loc, scale])

prob = normal_factor(
    {
        value: torch.tensor(0.0),
        loc: torch.tensor(0.0),
        scale: torch.tensor(1.0),
    }
)

# note to myself: this returns the probability, not log prob
print(prob)

# compare to the log prob from calling torch distribution directly
log_prob = dist.Normal(torch.tensor(0.0), torch.tensor(1.0)).log_prob(torch.tensor(0.0))
assert prob == log_prob.exp()
