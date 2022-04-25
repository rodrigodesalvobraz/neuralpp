from timeit import default_timer as timer
from typing import List, Dict, Any

import beanmachine.ppl as bm
import torch

from neuralpp.experiments.bm_integration.converter import BeanMachineConverter
from neuralpp.inference.graphical_model.representation.factor.factor import Factor
from neuralpp.inference.graphical_model.representation.factor.product_factor import ProductFactor
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import PyTorchTableFactor
from neuralpp.inference.graphical_model.representation.random.multi_type_random_model import MultiTypeRandomModel, \
    FactorMaker
from neuralpp.inference.graphical_model.representation.random.multi_type_random_model_util import \
    make_gaussian_with_mean, make_switch_of_gaussians_with_mean, make_randomly_shifted_standard_gaussian_given_range
from neuralpp.inference.graphical_model.variable.discrete_variable import DiscreteVariable
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.graphical_model.variable.tensor_variable import TensorVariable
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util import util
from neuralpp.util.util import empty

if __name__ == "__main__":

    print("Randomly generating model")
    #util.set_seed()
    util.set_seed(7665602960673477862)

    number_of_estimations = 50
    number_of_samples_per_estimation = 200
    number_of_chains = 1
    max_number_of_estimations_shown = 10
    assert number_of_estimations > 1, "We need at least 2 estimations for computing variance"

    model_size_multiplier = 10
    loop_coefficient = 0.9

    from_type_to_number_of_seed_variables = {
        IntegerVariable: 0,
        TensorVariable: 3,
    }

    ratio_of_total_number_of_variables_and_number_of_seed_variables = 5

    from_type_to_number_of_seed_variables = {type: size * model_size_multiplier
                                             for type, size in from_type_to_number_of_seed_variables.items()}

    threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary = \
        sum(from_type_to_number_of_seed_variables.values()) \
        * ratio_of_total_number_of_variables_and_number_of_seed_variables

    original_factors = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary=
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary,
            from_type_to_number_of_seed_variables=from_type_to_number_of_seed_variables,
            factor_makers=[
                FactorMaker([TensorVariable],
                            lambda variables: make_randomly_shifted_standard_gaussian_given_range(variables, range=10)),
                FactorMaker([TensorVariable, TensorVariable],
                            make_gaussian_with_mean),
                # repeating multiple times to increase chance of using switches
                FactorMaker([TensorVariable, IntegerVariable, TensorVariable, TensorVariable],
                            make_switch_of_gaussians_with_mean),
                FactorMaker([TensorVariable, IntegerVariable, TensorVariable, TensorVariable],
                            make_switch_of_gaussians_with_mean),
                FactorMaker([TensorVariable, IntegerVariable, TensorVariable, TensorVariable],
                            make_switch_of_gaussians_with_mean),
                FactorMaker([TensorVariable, IntegerVariable, TensorVariable, TensorVariable],
                            make_switch_of_gaussians_with_mean),
                FactorMaker([TensorVariable, IntegerVariable, TensorVariable, TensorVariable],
                            make_switch_of_gaussians_with_mean),
                FactorMaker([IntegerVariable],
                            lambda variables: PyTorchTableFactor(variables, [0.4, 0.6])),
                FactorMaker([IntegerVariable, IntegerVariable],
                            lambda variables: PyTorchTableFactor(variables, [[0.4, 0.3], [0.6, 0.7], ])),
            ],
            from_type_to_variable_maker={IntegerVariable: lambda name: IntegerVariable(name, 2)},
            loop_coefficient=loop_coefficient,
        ).from_variable_to_distribution.values()


    class Model:
        def __init__(self, name: str, factors: List[Factor]):
            self.name = name
            self.factors = factors
            self.variables = list(set(v for f in factors for v in f.variables))
            self.variables.sort(key=lambda v: str(v))
            self.discrete_variables = list(filter(lambda v: isinstance(v, DiscreteVariable), self.variables))
            self.continuous_variables = util.subtract(self.variables, self.discrete_variables)

        def print(self):
            print(util.join(self.factors, sep="\n"))
            print(f"{len(self.variables)} variables: {self.variables}")
            print(f"{len(self.discrete_variables)} discrete variables: {self.discrete_variables}")
            print(f"{len(self.continuous_variables)} continuous variables: {self.continuous_variables}")


    original_model = Model("Original", original_factors)

    marginalized_factors = ProductFactor.factors(ProductFactor(original_factors) ^ original_model.discrete_variables)
    marginalized_factors = [f for f in marginalized_factors if not empty(f.variables)]
    marginalized_model = Model("Marginalized", marginalized_factors)
    assert all(isinstance(v, TensorVariable) for f in marginalized_model.factors for v in f.variables)

    query = util.first(original_model.continuous_variables)
    variable_assignments: Dict[Variable, Any] = {}

    print()
    print("Original model:")
    original_model.print()

    print()
    print("Marginalized model:")
    marginalized_model.print()

    print(f"Query: {query}")

    print("Randomly generating samples")
    util.set_seed()  # Same model, different samples


    def solve_with_bm(model: Model, query: Variable, variable_assignments: Dict[Variable, Any]):
        converter = BeanMachineConverter(model.factors, variable_assignments)

        compositional = bm.CompositionalInference(
            {
                **{
                    converter.invoke_rv_function_of(
                        v
                    ).wrapper: bm.SingleSiteAncestralMetropolisHastings()
                    for v in model.discrete_variables
                },
                ...: bm.GlobalNoUTurnSampler(),
            }
        )
        queries = [converter.invoke_rv_function_of(query)]

        print()
        print(f"Starting inference on {model.name}")

        samples = compositional.infer(
            queries=queries,
            observations=converter.observations,
            num_samples=number_of_samples_per_estimation,
            num_adaptive_samples=number_of_samples_per_estimation // number_of_chains,
            num_chains=number_of_chains,
        )
        return samples

    def samples_mean(samples):
        return next(iter(samples.samples.values())).mean()

    def estimation_tensor_mean_and_variance(samples: List[torch.Tensor]):
        estimations = []
        for sample in samples:
            estimations.append(samples_mean(sample))
        estimation_tensor = torch.tensor(estimations)
        return estimation_tensor, estimation_tensor.mean(), estimation_tensor.var()


    models = [
        original_model,
        marginalized_model,
    ]

    tensor_of_estimations = [torch.tensor([]) for m in models]
    list_of_estimation_times = [[] for m in models]

    for estimation in range(number_of_estimations):
        print()
        print(f"Starting {estimation + 1}-th estimation")

        for model_index, model in enumerate(models):
            start = timer()
            estimation_samples = solve_with_bm(model, query, variable_assignments)
            end = timer()
            tensor_of_estimations[model_index] = \
                util.tensor1d_cat_value(tensor_of_estimations[model_index], samples_mean(estimation_samples))
            list_of_estimation_times[model_index].append(end - start)

        if estimation > 0:
            variances = []
            for model_index, model in enumerate(models):
                print(f"{model.name:13}")
                estimation_tensor = tensor_of_estimations[model_index]
                mean, variance = estimation_tensor.mean(), estimation_tensor.var()
                variances.append(variance)
                estimation_str = util.join([f"{e:.3f}" for e in estimation_tensor[-max_number_of_estimations_shown:]])
                estimations_description = "estimations" \
                    if len(estimation_tensor) <= max_number_of_estimations_shown \
                    else f"last {max_number_of_estimations_shown} estimations"
                print(f"    {estimations_description}: {estimation_str}")
                print(f"    mean {mean:.3f} +- {variance:.3f} variance")
            if min(variances) > 0:
                print(f"Max/min variance ratio: {max(variances)/min(variances):.3f}")
