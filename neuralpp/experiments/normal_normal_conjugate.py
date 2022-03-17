import torch
import torch.distributions as dist
import beanmachine.ppl as bm
import copy
from tqdm.auto import tqdm
from typing import NamedTuple

from neuralpp.inference.graphical_model.representation.factor.continuous.normal_factor import (
    NormalFactor,
)
from neuralpp.inference.graphical_model.variable.tensor_variable import TensorVariable
from neuralpp.experiments.bm_integration.factor_world import FactorWorld

if __name__ == "__main__":
    # Define Normal Normal in neuralpp

    # first Normal
    mu = TensorVariable("mu", 0)
    std = TensorVariable("std", 0)
    normal_1_out = TensorVariable("normal_1_out", 0)

    normal_1 = NormalFactor([normal_1_out, mu, std])

    # second Normal (note the reuse of normal_1_out)
    sigma = TensorVariable("sigma", 0)
    normal_2_val = TensorVariable("normal_2_val", 0)

    normal_2 = NormalFactor([normal_2_val, normal_1_out, sigma])

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
        normal_1_samples.append(world[normal_1_out])

    print(torch.stack(normal_1_samples).mean())

    # An equivalent BM model for reference
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

    model = NormalNormalModel(mu_val, std_val, sigma_val)
    samples = bm.GlobalNoUTurnSampler().infer(
        [model.normal_1()],
        {model.normal_2(): normal_2_obs},
        num_samples=num_samples,
        num_adaptive_samples=num_adaptive_samples,
        num_chains=1,
    )
    print(samples[model.normal_1()].mean())
