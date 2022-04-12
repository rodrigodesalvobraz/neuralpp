from typing import List, Dict, Any

import beanmachine.ppl as bm

from neuralpp.experiments.bm_integration.converter import BeanMachineConverter
from neuralpp.inference.graphical_model.representation.factor.factor import Factor
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import PyTorchTableFactor
from neuralpp.inference.graphical_model.representation.random.multi_type_random_model import MultiTypeRandomModel, \
    FactorMaker
from neuralpp.inference.graphical_model.representation.random.multi_type_random_model_util import \
    make_gaussian_with_mean, make_switch_of_gaussians_with_mean, make_standard_gaussian
from neuralpp.inference.graphical_model.variable.discrete_variable import DiscreteVariable
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable
from neuralpp.inference.graphical_model.variable.tensor_variable import TensorVariable
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util import util

if __name__ == "__main__":

    num_samples = 10000

    size_multiplier = 10
    loop_coefficient = 0.5

    neuralpp_model = \
        MultiTypeRandomModel(
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary=6 * size_multiplier,
            from_type_to_number_of_seed_variables={
                IntegerVariable: 1 * size_multiplier,
                TensorVariable: 3 * size_multiplier,
            },
            factor_makers=[
                FactorMaker([TensorVariable],
                            make_standard_gaussian),
                FactorMaker([TensorVariable, TensorVariable],
                            make_gaussian_with_mean),
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

    variable_assignments: Dict[Variable, Any] = {}

    variables = list(set(v for f in neuralpp_model for v in f.variables))
    discrete_variables = list(filter(lambda v: isinstance(v, DiscreteVariable), variables))
    continuous_variables = util.subtract(variables, discrete_variables)
    query = util.first(variables)

    converter = BeanMachineConverter(neuralpp_model, variable_assignments)

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
    samples = compositional.infer(
        queries=queries,
        observations=converter.observations,
        num_samples=num_samples,
        num_adaptive_samples=num_samples // 2,
        num_chains=2,
    )
