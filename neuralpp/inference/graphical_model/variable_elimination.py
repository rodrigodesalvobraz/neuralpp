from neuralpp.inference.graphical_model.representation.factor.product_factor import ProductFactor
from neuralpp.inference.graphical_model.variable.variable import Variable


def default_choose_next_to_eliminate(query_variables, product_of_remaining_factors):
    for factor in ProductFactor.factors(product_of_remaining_factors):
        for variable in factor.variables:
            if variable not in query_variables:
                return variable
    return None


class VariableElimination:

    def __init__(self, choose_next_to_eliminate=default_choose_next_to_eliminate):
        self.choose_next_to_eliminate = choose_next_to_eliminate

    def run(self, query, factors):
        """
        Returns a representation equivalent to sum_Others prod factors,
        where Others is all variables but query (which can be a single variable or collection thereof).
        """

        query_variables = [query] if isinstance(query, Variable) else query

        product_of_remaining_factors = ProductFactor(factors)
        while next_variable := self.choose_next_to_eliminate(query_variables, product_of_remaining_factors):
            product_of_remaining_factors = product_of_remaining_factors ^ next_variable

        return product_of_remaining_factors.normalize()
