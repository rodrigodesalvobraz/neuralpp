import torch
import torch.distributions as dist
import beanmachine.ppl as bm
import copy
import tqdm

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
        self._fixed_assignment_dict = fixed_assignment_dict

        # collect variables that need to be inferred
        variables = set()
        for factor in factors:
            variables = variables.union(factor.variables)
        unknown_variables = variables - fixed_assignment_dict.keys()

        # initialize values (this will be updated by inference algorithms)
        self._values = dict()
        for variable in unknown_variables:
            # some dummy initialization to get this script running. This does not work
            # in general for distribution with limited support
            self._values[variable] = dist.Uniform(-2, 2).sample()

    def __getitem__(self, variable):
        return self._values[variable]

    def replace(self, assignment_dict):
        # return a new world with updated values
        new_world = copy.copy(self)
        new_world._values = assignment_dict
        return new_world

    def log_prob(self):
        # return the log prob of the entire graph conditioned on the current value
        all_assignments = {**self._fixed_assignment_dict, **self._values}
        log_prob = 0.0
        for factor in self._factors:
            # evaluate each factor on the assignments
            log_prob += factor(all_assignments).log()
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

# the proposer is for internal use only, but let's see if we can get it work for our
# purpose...
nuts_proposer = bm.inference.proposer.nuts_proposer.NUTSProposer(
    initial_world,
    target_rvs=initial_world._values.keys(),
    num_adaptive_sample=num_adaptive_samples,
)


# begin inference
world = initial_world
normal_1_samples = [world[normal_1_val]]
for i in tqdm.trange(num_samples):
    world, _ = nuts_proposer.propose(world)

    # we only need to manually invoke these methods because we're using a BM internal
    # class :P
    if i < num_adaptive_samples:
        nuts_proposer.do_adaptation()
    if i == num_adaptive_samples - 1:
        nuts_proposer.finish_adaptation()

    normal_1_samples.append(world[normal_1_val])

print(torch.stack(normal_1_samples))

# An equivalent BM run for reference
model = NormalNormalModel(mu_val, std_val, sigma_val)
samples = bm.GlobalNoUTurnSampler().infer(
    [model.normal_1()],
    {model.normal_2(): normal_2_obs},
    num_samples=num_samples,
    num_adaptive_samples=num_adaptive_samples,
    num_chains=1,
)
print(samples)
