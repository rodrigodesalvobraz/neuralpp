import torch
import torch.distributions as dist
import beanmachine.ppl as bm
import copy
from tqdm.auto import tqdm

from neuralpp.inference.graphical_model.representation.factor.continuous.normal_factor import (
    NormalFactor,
)
from neuralpp.inference.graphical_model.variable.tensor_variable import TensorVariable

# Equivalent BM model for reference
class NormalNormalModel:
    def __init__(self, mu: torch.Tensor, std: torch.Tensor, sigma: torch.Tensor):
        self.mu_ = mu
        self.std_ = std
        self.sigma_ = sigma

    @bm.random_variable
    def normal_1(self):
        return dist.Normal(self.mu_, self.std_)

    @bm.random_variable
    def normal_2(self):
        return dist.Normal(self.normal_1(), self.sigma_)


# Define Normal Normal in neuralpp

# first Normal
mu = TensorVariable("mu", 0)
std = TensorVariable("std", 0)
normal_1_val = TensorVariable("normal_1_val", 0)

normal_1 = NormalFactor([normal_1_val, mu, std])

# second Normal (note the reuse of normal_1_val)
sigma = TensorVariable("sigma", 0)
normal_2_val = TensorVariable("normal_2_val", 0)

normal_2 = NormalFactor([normal_2_val, normal_1_val, sigma])

# a "World" that explains factor graph to BM inference methods


class FactorWorld(bm.world.World):
    def __init__(self, factors, fixed_assignment_dict):
        self._factors = factors
        # for variables whose values are fixed during an inference
        self.observations = fixed_assignment_dict.copy()

        # collect variables that need to be inferred
        variables = set()
        for factor in factors:
            variables = variables.union(factor.variables)

        # initialize values (this will be updated by inference algorithms)
        self._variables = dict()
        for var in variables:
            if var in self.observations:
                self._variables[var] = self.observations[var]
            else:
                # some dummy initialization to get this script running. This does not work
                # in general for distribution with limited support
                self._variables[var] = dist.Uniform(-2, 2).sample()

    def __getitem__(self, variable):
        return self._variables[variable]

    def replace(self, assignment_dict):
        # return a new world with updated values
        new_world = copy.copy(self)
        new_world._variables = {**self._variables, **assignment_dict}
        return new_world

    def log_prob(self, factors=None):
        # return the log prob of the factors conditioned on the current value
        log_prob = 0.0

        # return log prob of entire graph if not provided
        if factors is None:
            factors = self._factors

        for factor in factors:
            # evaluate each factor on the assignments
            log_prob += factor(self._variables).log()
        return log_prob

    def get_variable(self, variable):
        # dummy function to bypass HMC's check on distributions' support
        # https://github.com/facebookresearch/beanmachine/blob/main/src/beanmachine/ppl/inference/proposer/hmc_utils.py#L257
        class DummyVar:
            distribution = dist.Normal(0.0, 1.0)

        return DummyVar()


# define fixed hyperparameter
mu_val = torch.tensor(10.0)
std_val = torch.tensor(2.0)
sigma_val = torch.tensor(5.0)

# define observation
normal_2_obs = torch.tensor(15.9)


factors = [normal_1, normal_2]
fixed_assignments = {
    mu: mu_val,
    std: std_val,
    sigma: sigma_val,
    normal_2_val: normal_2_obs,
}


initial_world = FactorWorld(factors, fixed_assignments)


num_samples = 200
num_adaptive_samples = num_samples // 2

# we usually don't manually construct the sampler object, but let's just try to see
# if it's possible to get it working here...
sampler = bm.inference.sampler.Sampler(
    kernel=bm.GlobalNoUTurnSampler(),
    initial_world=initial_world,
    num_samples=num_samples,
    num_adaptive_samples=num_adaptive_samples,
)


# begin inference
normal_1_samples = []
for world in tqdm(sampler, total=num_samples + num_adaptive_samples):
    normal_1_samples.append(world[normal_1_val])

print(torch.stack(normal_1_samples).mean())

# An equivalent BM run for reference
model = NormalNormalModel(mu_val, std_val, sigma_val)
samples = bm.GlobalNoUTurnSampler().infer(
    [model.normal_1()],
    {model.normal_2(): normal_2_obs},
    num_samples=num_samples,
    num_adaptive_samples=num_adaptive_samples,
    num_chains=1,
)
print(samples[model.normal_1()].mean())
