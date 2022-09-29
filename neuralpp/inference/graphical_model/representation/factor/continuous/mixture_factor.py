from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
)
from neuralpp.inference.graphical_model.variable.integer_variable import (
    DiscreteVariable,
)


class MixtureFactor(Factor):
    def __init__(
        self, marginalized_variable: DiscreteVariable, factor: Factor
    ):
        if not isinstance(marginalized_variable, DiscreteVariable):
            raise ValueError(
                "Only (finite) discrete variables (with 'cardinality' property) can be marginalized."
            )
        super().__init__(
            [v for v in factor.variables if v != marginalized_variable]
        )

        self.raw_factor = factor
        self.marginalized_variable = marginalized_variable

    def condition_on_non_empty_dict(self, assignment_dict):
        return MixtureFactor(
            self.marginalized_variable,
            self.raw_factor.condition(assignment_dict),
        )

    def call_after_validation(self, assignment_dict, assignment_values):
        prob = 0.0
        assignment_dict = assignment_dict.copy()
        for val in self.marginalized_variable.assignments():
            assignment_dict[self.marginalized_variable] = val
            prob += self.raw_factor(assignment_dict)
        return prob

    def mul_by_non_identity(self, other: Factor):
        if self.marginalized_variable in other.variables:
            raise ValueError(
                f"{other} contains a variable that has been marginalized out."
            )

        product = self.raw_factor * other
        return MixtureFactor(self.marginalized_variable, product)

    def sum_out_variable(self, variable):
        if variable == self.marginalized_variable:
            return self
        reduced = self.raw_factor.sum_out_variable(variable)
        return MixtureFactor(self.marginalized_variable, reduced)

    def __repr__(self):
        return f"MixtureFactor({repr(self.marginalized_variable)}, {repr(self.factor)})"

    def __str__(self):
        return f"Mixture factor based on {self.marginalized_variable} and {self.raw_factor}"
