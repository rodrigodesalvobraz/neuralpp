import datetime
from timeit import default_timer as timer
from typing import List, Dict, Any, Tuple

import beanmachine.ppl as bm
import matplotlib.pyplot as plt
import torch

from neuralpp.experiments.bm_integration.converter import BeanMachineConverter
from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
)
from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    ProductFactor,
)
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.factor.switch_factor import (
    SwitchFactor,
)
from neuralpp.inference.graphical_model.representation.random.multi_type_random_model import (
    MultiTypeRandomModel,
    FactorMaker,
)
from neuralpp.inference.graphical_model.representation.random.multi_type_random_model_util import (
    make_gaussian_with_mean,
    make_switch_of_gaussians_with_mean,
    make_randomly_shifted_standard_gaussian_given_range,
    make_random_table_factor,
    make_shifted_standard_gaussian_given_shift,
)
from neuralpp.inference.graphical_model.variable.discrete_variable import (
    DiscreteVariable,
)
from neuralpp.inference.graphical_model.variable.integer_variable import (
    IntegerVariable,
)
from neuralpp.inference.graphical_model.variable.tensor_variable import (
    TensorVariable,
)
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util import util
from neuralpp.util.util import empty, list_for_each


def main():
    print("Started at:", datetime.datetime.now().strftime("%Y-%mv-%d %H:%M:%S"))

    print("Randomly generating model")
    util.set_seed()
    # util.set_seed(7665602960673477862)

    number_of_generated_models = 20

    # number of samples to be used per estimation of variance
    initial_number_of_samples = 500
    final_number_of_samples = 5000
    number_of_samples_step = 500

    number_of_estimations = 50

    number_of_chains = 1
    max_number_of_estimations_shown = 10
    assert (
        number_of_estimations > 1
    ), "We need at least 2 estimations for computing variance"
    model_size_multiplier = 1
    loop_coefficient = 0.9
    from_type_to_number_of_seed_variables = {
        IntegerVariable: 0,
        TensorVariable: 3,
    }

    cardinality_of_discrete_variables = 5

    ratio_of_total_number_of_variables_and_number_of_seed_variables = 5

    numbers_of_samples = range(
        initial_number_of_samples,
        final_number_of_samples + number_of_samples_step,
        number_of_samples_step,
    )

    from_type_to_number_of_seed_variables = {
        type: size * model_size_multiplier
        for type, size in from_type_to_number_of_seed_variables.items()
    }
    threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary = (
        sum(from_type_to_number_of_seed_variables.values())
        * ratio_of_total_number_of_variables_and_number_of_seed_variables
    )

    types_of_variables_in_gaussians_in_switch = [
        TensorVariable
    ] * cardinality_of_discrete_variables

    types_of_variables_in_switch_of_gaussians = [
        TensorVariable,
        IntegerVariable,
    ] + types_of_variables_in_gaussians_in_switch

    ######

    def samples_mean(samples):
        return next(iter(samples.samples.values())).mean()

    class Model:
        def __init__(self, name: str, factors: List[Factor]):
            self.name = name
            self.factors = factors
            self.variables = list(set(v for f in factors for v in f.variables))
            self.variables.sort(key=lambda v: str(v))
            self.discrete_variables = list(
                filter(lambda v: isinstance(v, DiscreteVariable), self.variables)
            )
            self.continuous_variables = util.subtract(
                self.variables, self.discrete_variables
            )

        def print(self):
            print(util.join(self.factors, sep="\n"))
            print(f"{len(self.variables)} variables: {self.variables}")
            print(
                f"{len(self.discrete_variables)} discrete variables: {self.discrete_variables}"
            )
            print(
                f"{len(self.continuous_variables)} continuous variables: {self.continuous_variables}"
            )

    def solve_with_bm(
        model: Model,
        query: Variable,
        variable_assignments: Dict[Variable, Any],
        number_of_samples_per_estimation: int,
        number_of_chains: int,
    ):
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

    ##########

    def generate_original_factors():
        nonlocal cardinality_of_discrete_variables
        original_factors = MultiTypeRandomModel(
            threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary=threshold_number_of_variables_to_avoid_new_variables_unless_absolutely_necessary,
            from_type_to_number_of_seed_variables=from_type_to_number_of_seed_variables,
            factor_makers=[
                FactorMaker(
                    [TensorVariable],
                    lambda variables: make_randomly_shifted_standard_gaussian_given_range(
                        variables, range=10
                    ),
                ),
                FactorMaker([TensorVariable, TensorVariable], make_gaussian_with_mean),
                # repeating multiple times to increase chance of using switches
                FactorMaker(
                    types_of_variables_in_switch_of_gaussians,
                    make_switch_of_gaussians_with_mean,
                ),
                FactorMaker(
                    types_of_variables_in_switch_of_gaussians,
                    make_switch_of_gaussians_with_mean,
                ),
                FactorMaker(
                    types_of_variables_in_switch_of_gaussians,
                    make_switch_of_gaussians_with_mean,
                ),
                FactorMaker(
                    types_of_variables_in_switch_of_gaussians,
                    make_switch_of_gaussians_with_mean,
                ),
                FactorMaker(
                    types_of_variables_in_switch_of_gaussians,
                    make_switch_of_gaussians_with_mean,
                ),
                FactorMaker([IntegerVariable], make_random_table_factor),
                FactorMaker(
                    [IntegerVariable, IntegerVariable],
                    make_random_table_factor,
                ),
            ],
            from_type_to_variable_maker={
                IntegerVariable: lambda name: IntegerVariable(
                    name, cardinality_of_discrete_variables
                )
            },
            loop_coefficient=loop_coefficient,
        ).from_variable_to_distribution.values()

        use_simple_mixture_of_gaussians_instead = True
        if use_simple_mixture_of_gaussians_instead:
            cardinality_of_discrete_variables = 5
            distance_between_means = 2
            index = IntegerVariable("index", cardinality_of_discrete_variables)
            y = TensorVariable("y")
            components = [
                make_shifted_standard_gaussian_given_shift(
                    [y], distance_between_means * i
                )
                for i in range(cardinality_of_discrete_variables)
            ]
            original_factors = [
                PyTorchTableFactor(
                    [index],
                    [1.0 / cardinality_of_discrete_variables]
                    * cardinality_of_discrete_variables,
                ),
                SwitchFactor(index, components),
            ]
        return original_factors

    def make_model_variants_and_query(
        original_factors,
    ) -> Tuple[Tuple[Model, Model], Variable]:
        original_model = Model("Original", original_factors)
        marginalized_factors = ProductFactor.factors(
            ProductFactor(original_factors) ^ original_model.discrete_variables
        )
        marginalized_factors = [
            f for f in marginalized_factors if not empty(f.variables)
        ]
        marginalized_model = Model("Marginalized", marginalized_factors)
        assert all(
            isinstance(v, TensorVariable)
            for f in marginalized_model.factors
            for v in f.variables
        )
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
        return (original_model, marginalized_model), query

    def generate_model_variants_and_query() -> Tuple[Tuple[Model, Model], Variable]:
        return make_model_variants_and_query(generate_original_factors())

    model_variants_and_queries_per_generated_model = [
        generate_model_variants_and_query() for _ in range(number_of_generated_models)
    ]

    util.set_seed()  # Same model, different samples

    def get_estimate_and_time(model, query, number_of_samples):
        start = timer()
        variable_assignments: Dict[Variable, Any] = {}
        estimation_samples = solve_with_bm(
            model,
            query,
            variable_assignments,
            number_of_samples,
            number_of_chains,
        )
        mean_estimate = samples_mean(estimation_samples)
        end = timer()
        time = end - start
        return mean_estimate, time

    data = list_for_each(
        numbers_of_samples,
        lambda number_of_samples: list_for_each(
            range(number_of_estimations),
            pre=lambda estimation: print(
                f"\nStarting {estimation + 1}-th estimation "
                f"(out of {number_of_estimations}) "
                f"with {number_of_samples} samples"
            ),
            body=lambda estimation: list_for_each(
                model_variants_and_queries_per_generated_model[0][0],
                lambda model_variant: get_estimate_and_time(
                    model=model_variant,
                    query=model_variants_and_queries_per_generated_model[0][1],
                    number_of_samples=number_of_samples,
                ),
            ),
            post_index_result=lambda estimation, data_by_model_by_estimation: ongoing_summary(
                estimation,
                data_by_model_by_estimation,
                max_number_of_estimations_shown,
                model_variants_and_queries_per_generated_model[0][0],
            ),
        ),
    )

    assert len(data) == len(numbers_of_samples)
    assert len(data[0]) == number_of_estimations
    assert len(data[0][0]) == len(model_variants_and_queries_per_generated_model[0][0])

    def aggregate_data_over_estimations(tensor_aggregation_function, data_field_index):
        return [
            [
                tensor_aggregation_function(
                    torch.tensor(
                        [
                            data[ns][e][mv][data_field_index]
                            for e in range(number_of_estimations)
                        ]
                    )
                )
                for mv, _ in enumerate(
                    model_variants_and_queries_per_generated_model[0][0]
                )
            ]
            for ns, _ in enumerate(numbers_of_samples)
        ]

    variance_by_model_by_number_of_samples = aggregate_data_over_estimations(
        torch.Tensor.var, 0
    )  # 0 = estimate
    mean_time_by_model_by_number_of_samples = aggregate_data_over_estimations(
        torch.Tensor.mean, 1
    )  # 1 = time

    print(f"Variances: {variance_by_model_by_number_of_samples}")
    print(f"Times    : {mean_time_by_model_by_number_of_samples}")

    colors = ["red", "blue"]
    for model_index in range(len(model_variants_and_queries_per_generated_model[0][0])):
        variance_by_number_of_samples = [
            variance_by_model[model_index]
            for variance_by_model in variance_by_model_by_number_of_samples
        ]
        mean_time_by_number_of_samples = [
            mean_time_by_model[model_index]
            for mean_time_by_model in mean_time_by_model_by_number_of_samples
        ]
        plt.plot(
            mean_time_by_number_of_samples,
            variance_by_number_of_samples,
            color=colors[model_index],
        )
    plt.show()

    print("Finished at:", datetime.datetime.now().strftime("%Y-%mv-%d %H:%M:%S"))


def ongoing_summary(
    current_estimation,
    data_by_model_by_estimation,
    max_number_of_estimations_shown,
    models,
):
    if current_estimation > 0:
        variance_by_model = []
        for model_index, model in enumerate(models):
            print(f"{model.name:13}")
            estimates_so_far = [
                data_by_model_by_estimation[e][model_index][0]
                for e in range(current_estimation + 1)
            ]
            tensor_of_estimates = torch.tensor(estimates_so_far)
            mean_of_estimates = tensor_of_estimates.mean()
            variance_of_estimates = tensor_of_estimates.var()
            estimates_to_show = tensor_of_estimates[-max_number_of_estimations_shown:]
            estimates_str = util.join([f"{e1 :.3f}" for e1 in estimates_to_show])
            estimates_description = (
                "variance_of_estimates"
                if len(tensor_of_estimates) <= max_number_of_estimations_shown
                else f"last {max_number_of_estimations_shown} variance_of_estimates"
            )
            print(f"    {estimates_description}: {estimates_str}")
            print(
                f"    mean {mean_of_estimates:.3f} +- {variance_of_estimates :.3f} variance"
            )
            variance_by_model.append(variance_of_estimates)
        if min(variance_by_model) > 0:
            print(
                f"Max/min variance ratio: "
                f"{max(variance_by_model) / min(variance_by_model):.3f}"
            )


if __name__ == "__main__":
    main()
