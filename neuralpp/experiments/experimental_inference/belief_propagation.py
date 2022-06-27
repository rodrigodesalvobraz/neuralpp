from typing import List, Optional

from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    Factor, ProductFactor
)
from neuralpp.inference.graphical_model.variable.variable import Variable


class BeliefPropagation:

    @staticmethod
    def run(query: Variable, factors: List[Factor]):
        return BeliefPropagation.msg_from_variable(query, None, factors).normalize()

    @staticmethod
    def msg_from_factor(factor: Factor, out_var: Variable, factors: List[Factor]):
        neighbors = [v for v in factor.variables if v != out_var]
        if not neighbors:
            return factor
        return ProductFactor(
            [factor] + [BeliefPropagation.msg_from_variable(v, factor, factors)
                        for v in neighbors]
        ).sum_out_variables(neighbors)

    @staticmethod
    def msg_from_variable(variable: Variable, out_fac: Optional[Factor], factors: List[Factor]):
        neighbors = [f for f in factors if variable in f.variables and (out_fac is None or f != out_fac)]
        msgs_in = [BeliefPropagation.msg_from_factor(f, variable, factors) for f in neighbors]
        return ProductFactor(msgs_in).atomic_factor()
