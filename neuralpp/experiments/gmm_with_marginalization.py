from torch.distributions.normal import Normal
from neuralpp.inference.graphical_model.variable.tensor_variable import TensorVariable
from neuralpp.inference.graphical_model.representation.factor.continuous.normal_factor import (
    NormalFactor,
)
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    ProductFactor,
)
from neuralpp.inference.graphical_model.representation.factor.switch_factor import (
    SwitchFactor,
)
from neuralpp.experiments.normal_normal_conjugate import FactorWorld
import beanmachine.ppl as bm
from tqdm.auto import tqdm
import torch


if __name__ == "__main__":
    K = 2  # number of components
    data = torch.tensor([0.0, 1.0, 10.0, 11.0, 12.0])  # a tiny dataset of 5 points

    # for simplicity (since we haven't implemented other distributions yet), assume
    # uniform cluster size and a fixed std of 2.0 for each cluster
    cluster_weight = torch.ones((K,)) / K

    std = TensorVariable("std", 0)
    std_data = torch.tensor(2.0)

    # mu(k) ~ Normal(0.0, 10.0)
    mu_loc = TensorVariable("mu_loc", 0)
    mu_loc_data = torch.tensor(0.0)
    mu_std = TensorVariable("mu_std", 0)
    mu_std_data = torch.tensor(10.0)
    mu_out = []
    mu_factors = []
    for k in range(K):
        # construct each cluster
        mu_out.append(TensorVariable(f"mu_out_{k}", 0))
        mu_factors.append(NormalFactor([mu_out[k], mu_loc, mu_std]))

    assignments = []
    assignment_factors = []
    obs = []
    obs_factors = []
    for n in range(len(data)):
        assignments.append(IntegerVariable(f"assignment_{n}", K))
        assignment_factors.append(PyTorchTableFactor([assignments[n]], cluster_weight))

        obs.append(TensorVariable(f"obs_{n}", 0))
        components = []
        for k in range(K):
            components.append(NormalFactor([obs[n], mu_out[k], std]))
        obs_factors.append(SwitchFactor(assignments[n], components))

    # collect all factors
    factors = [*mu_factors, *assignment_factors, *obs_factors]
    marginalized_factor = ProductFactor(factors) ^ assignments
    print(marginalized_factor)

    observations = {
        std: std_data,
        mu_loc: mu_loc_data,
        mu_std: mu_std_data,
        **{obs[n]: data[n] for n in range(len(data))},
    }

    # begin building the World
    initial_world = FactorWorld([marginalized_factor], observations)

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
    mu_samples = {mu: [] for mu in mu_out}
    for world in tqdm(sampler, total=num_samples + num_adaptive_samples):
        # collect the value of mu (the centroid of each component)
        for mu in mu_out:
            mu_samples[mu].append(world[mu])
    mu_samples_mean = {key: torch.stack(val).mean() for key, val in mu_samples.items()}

    print(mu_samples_mean)
