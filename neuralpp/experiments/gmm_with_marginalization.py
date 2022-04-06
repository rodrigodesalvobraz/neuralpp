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
from neuralpp.experiments.bm_integration.converter import BeanMachineConverter
from neuralpp.experiments.bm_integration.benchmark import benchmark
import beanmachine.ppl as bm
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

    zs = []
    z_factors = []
    obs = []
    obs_factors = []
    for n in range(len(data)):
        zs.append(IntegerVariable(f"z_{n}", K))
        z_factors.append(PyTorchTableFactor([zs[n]], cluster_weight))

        obs.append(TensorVariable(f"obs_{n}", 0))
        components = []
        for k in range(K):
            components.append(NormalFactor([obs[n], mu_out[k], std]))
        obs_factors.append(SwitchFactor(zs[n], components))

    num_samples = 3000
    num_adaptive_samples = num_samples // 2
    variable_assignments = {
        std: std_data,
        mu_loc: mu_loc_data,
        mu_std: mu_std_data,
        **{obs[n]: data[n] for n in range(len(data))},
    }
    factors = [*mu_factors, *z_factors, *obs_factors]

    ######################################
    # Marginalization
    ######################################
    # collect all factors
    marginalized_factor = ProductFactor(factors) ^ zs
    print("marginalized_factor:", marginalized_factor)

    # converting to BM
    converter = BeanMachineConverter([marginalized_factor], variable_assignments)

    # begin inference
    queries = [converter.invoke_rv_function_of(mu) for mu in mu_out]
    # samples = bm.GlobalNoUTurnSampler().infer(
    #     queries=queries,
    #     observations=converter.observations,
    #     num_samples=num_samples,
    #     num_adaptive_samples=num_adaptive_samples,
    #     num_chains=1,
    # )

    # Benchmark marginalization
    marginalized_result = benchmark(
        infer_class=bm.GlobalNoUTurnSampler(),
        queries=queries,
        observations=converter.observations,
        num_samples=num_samples,
    )
    print(marginalized_result)

    ######################################
    # Compositional Inference
    ######################################
    converter = BeanMachineConverter(factors, variable_assignments)

    # begin inference

    # discrete variables (z) => Single site ancestral
    # rest of variables (...) => NUTS
    compositional = bm.CompositionalInference(
        {
            **{
                converter.invoke_rv_function_of(z).wrapper: bm.SingleSiteAncestralMetropolisHastings()
                for z in zs
            },
            ...: bm.GlobalNoUTurnSampler(),
        }
    )
    queries = [converter.invoke_rv_function_of(mu) for mu in mu_out]
    # samples = compositional.infer(
    #     queries=queries,
    #     observations=converter.observations,
    #     num_samples=num_samples,
    #     num_adaptive_samples=num_adaptive_samples,
    #     num_chains=1,
    # )

    # Benchmark compositional inference
    marginalized_result = benchmark(
        infer_class=compositional,
        queries=queries,
        observations=converter.observations,
        num_samples=num_samples,
    )
    print(marginalized_result)

    # TODO: plotting
