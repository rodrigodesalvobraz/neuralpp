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

    util.set_seed(None)

    number_of_samples = 3_000
    number_of_chains = 2
    number_of_estimations = 100

    model_size_multiplier = 5
    loop_coefficient = 0.5

    from_type_to_number_of_seed_variables = {
        IntegerVariable: 0,
        TensorVariable: 10,
    }

    ratio_of_total_number_of_variables_and_number_of_seed_variables = 3

    from_type_to_number_of_seed_variables = {type: size * model_size_multiplier
                                             for type, size in from_type_to_number_of_seed_variables.items()}

    threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary = \
        sum(from_type_to_number_of_seed_variables.values()) \
        * ratio_of_total_number_of_variables_and_number_of_seed_variables

    neuralpp_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary=
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary,
            from_type_to_number_of_seed_variables=from_type_to_number_of_seed_variables,
            factor_makers=[
                FactorMaker([TensorVariable],
                            lambda variables: make_randomly_shifted_standard_gaussian_given_range(variables, range=10)),
                FactorMaker([TensorVariable, TensorVariable],
                            make_gaussian_with_mean),
                # repeating multiple times to increase change of using switches
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

    print("Model:")
    print(util.join(neuralpp_model, sep="\n"))

    def solve_with_bm(model: List[Factor], query: Variable, variable_assignments: Dict[Variable, Any]):
        variables = list(set(v for f in model for v in f.variables))
        discrete_variables = list(filter(lambda v: isinstance(v, DiscreteVariable), variables))

        converter = BeanMachineConverter(model, variable_assignments)

        compositional = bm.CompositionalInference(
            {
                **{
                    converter.invoke_rv_function_of(
                        v
                    ).wrapper: bm.SingleSiteAncestralMetropolisHastings()
                    for v in discrete_variables
                },
                ...: bm.GlobalNoUTurnSampler(),
            }
        )
        queries = [converter.invoke_rv_function_of(query)]

        print(f"Starting inference on {len(variables)} ({len(discrete_variables)} discrete ones)")

        samples = compositional.infer(
            queries=queries,
            observations=converter.observations,
            num_samples=number_of_samples,
            num_adaptive_samples=number_of_samples // number_of_chains,
            num_chains=number_of_chains,
        )
        return samples


    variables = list(set(v for f in neuralpp_model for v in f.variables))
    discrete_variables = list(filter(lambda v: isinstance(v, DiscreteVariable), variables))
    continuous_variables = util.subtract(variables, discrete_variables)

    query = util.first(continuous_variables)
    variable_assignments: Dict[Variable, Any] = {}

    compositional_samples = [solve_with_bm(neuralpp_model, query, variable_assignments)
                             for i in range(number_of_estimations)]

    marginalized_model = ProductFactor.factors(ProductFactor(neuralpp_model) ^ discrete_variables)
    marginalized_model = [f for f in marginalized_model if not empty(f.variables)]
    print("\nMarginalized model:")
    print(util.join(marginalized_model, sep="\n"))
    assert all(isinstance(v, TensorVariable) for f in marginalized_model for v in f.variables)

    marginalized_samples = [solve_with_bm(marginalized_model, query, variable_assignments)
                            for i in range(number_of_estimations)]

    cs_variance = torch.tensor([next(iter(cs.samples.values())).mean() for cs in compositional_samples]).var()
    print(f"Compositional variance: {cs_variance}")

    ms_variance = torch.tensor([next(iter(ms.samples.values())).mean() for ms in marginalized_samples]).var()
    print(f"Marginalized variance: {ms_variance}")
