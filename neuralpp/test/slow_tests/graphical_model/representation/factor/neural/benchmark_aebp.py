import math
import random

from neuralpp.experiments.experimental_inference.approximations import message_approximation
from neuralpp.experiments.experimental_inference.exact_belief_propagation import AnytimeExactBeliefPropagation
from neuralpp.inference.graphical_model.representation.factor.factor import Factor
from neuralpp.inference.graphical_model.representation.random.random_model import generate_model


def main():
    def expansion_value_function_1(x, partial_tree, full_tree):
        # Essentially random prioritization
        return hash(id(x))

    def expansion_value_function_2(x, partial_tree, full_tree):
        # Prioritize nodes with the fewest children
        return -len(full_tree.children(x))

    def expansion_value_function_3(x, partial_tree, full_tree):
        # Prioritize higher-entropy factors
        if isinstance(x, Factor):
            table = x.normalize().table_factor.table.potentials_tensor()
            return -(table * table.log()).sum()
        return 0.5  # just some constant for variables, but perhaps we should treat this differently.

    def compute_all_approximations_aebp(factors, query):
        results = []
        aebp = AnytimeExactBeliefPropagation.from_factors(
            factors,
            query,
            expansion_value_function=expansion_value_function_3,
            approximation=message_approximation
        )
        results.append(aebp.run())
        while not aebp.is_complete():
            aebp.expand(query)
            results.append(aebp.run())
        return results

    def kl_divergence(table_factor_approximation, true_table_factor):
        p = true_table_factor.table
        q = table_factor_approximation.table
        return sum(p[i] * math.log(p[i] / q[i]) for i in range(len(p)))

    model = generate_model(
        number_of_factors=100, number_of_variables=15, cardinality=3
    )
    query_variable = random.choice([v for f in model for v in f.variables])
    result = compute_all_approximations_aebp(model, query_variable)
    final = result[-1]
    # kl_divergences = []
    for r in result:
        # kl_divergences.append(kl_divergence(r, final))
        print(kl_divergence(r, final).item())


if __name__ == "__main__":
    main()
