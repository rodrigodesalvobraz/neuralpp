from torch.distributions.normal import Normal
from neuralpp.inference.graphical_model.variable.tensor_variable import TensorVariable
from neuralpp.inference.graphical_model.representation.factor.continuous.normal_factor import (
    NormalFactor,
)
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    ProductFactor,
)
from neuralpp.inference.graphical_model.representation.factor.switch_factor import (
    SwitchFactor,
)
from neuralpp.experiments.bm_integration.factor_world import FactorWorld
from neuralpp.experiments.bm_integration.factor_to_rv import (
    make_random_variable,
    rv_functions,
    make_functional,
)
import beanmachine.ppl as bm
from beanmachine.ppl.inference.base_inference import BaseInference
from tqdm.auto import tqdm
import torch
import random
from typing import Dict, Tuple, Union


# It's hard to get BM's native CompositionalInference to work with factor
# representations, but we can implement our own "compositional inference for
# neuralpp"


class FactorCompositionalInference(BaseInference):
    def __init__(
        self, var_to_inference: Dict[Tuple[Variable, ...], BaseInference]
    ) -> None:
        self._config = var_to_inference

    def get_proposers(self, world, target_rvs, num_adaptive_sample):
        proposers = []
        covered_vars = set()
        for variables, inference_method in self._config.items():
            variables = set(var for var in variables if var in target_rvs)
            covered_vars |= variables
            proposers.extend(
                inference_method.get_proposers(world, variables, num_adaptive_sample)
            )

        remaining_vars = target_rvs - covered_vars
        # use ancestral MH for variables not specified
        proposers.extend(
            bm.SingleSiteAncestralMetropolisHastings().get_proposers(
                world, remaining_vars, num_adaptive_sample
            )
        )
        return proposers


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

    num_samples = 200
    num_adaptive_samples = num_samples // 2
    observations = {
        std: std_data,
        mu_loc: mu_loc_data,
        mu_std: mu_std_data,
        **{obs[n]: data[n] for n in range(len(data))},
    }
    factors = [*mu_factors, *assignment_factors, *obs_factors]

    ######################################
    # Marginalization
    ######################################
    # collect all factors
    marginalized_factor = ProductFactor(factors) ^ assignments
    print(marginalized_factor)

    # begin building the World
    initial_world = FactorWorld([marginalized_factor], observations)

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

    print("Result for marginalization:", mu_samples_mean)

    ######################################
    # Compositional Inference
    ######################################
    compositional_world = FactorWorld(factors, observations)
    compositional_sampler = bm.inference.sampler.Sampler(
        kernel=FactorCompositionalInference(
            {
                tuple(mu_out): bm.GlobalNoUTurnSampler(),
                # rest of variables fall back to default MH sampler
            }
        ),
        initial_world=compositional_world,
        num_samples=num_samples,
        num_adaptive_samples=num_adaptive_samples,
    )

    # begin inference
    mu_samples = {mu: [] for mu in mu_out}
    for world in tqdm(compositional_sampler, total=num_samples + num_adaptive_samples):
        # collect the value of mu (the centroid of each component)
        for mu in mu_out:
            mu_samples[mu].append(world[mu])
    mu_samples_mean = {key: torch.stack(val).mean() for key, val in mu_samples.items()}

    print("Result for Compositional Inference:", mu_samples_mean)

    ######################################
    # Random Variable wrapper (Compositional Inference)
    ######################################
    # rv_functions = {id(factor): }

    # create rv functions for factors
    for factor in factors:
        make_random_variable(factor)

    # for fixed priors
    make_functional(std, std_data)
    make_functional(mu_loc, mu_loc_data)
    make_functional(mu_std, mu_std_data)

    # print(rv_functions)

    queries = [rv_functions[mu]() for mu in mu_out]
    observations = {rv_functions[obs[n]](): data[n] for n in range(len(obs))}
    samples = bm.CompositionalInference(
        {
            **{
                rv_functions[assignment]: bm.SingleSiteAncestralMetropolisHastings()
                for assignment in assignments
            },
            ...: bm.GlobalNoUTurnSampler(),
        }
    ).infer(
        queries=queries,
        observations=observations,
        num_samples=num_samples,
        num_adaptive_samples=num_adaptive_samples,
        num_chains=1,
    )

    print(
        "Result from implementing the wrapper:",
        {query: samples[query].mean() for query in queries},
    )
