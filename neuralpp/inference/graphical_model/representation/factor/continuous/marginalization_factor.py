from neuralpp.inference.graphical_model import variable
from neuralpp.inference.graphical_model.representation.factor.continuous.continuous_internal_parameterless_factor import (
    ContinuousInternalParameterlessFactor,
)
from neuralpp.inference.graphical_model.variable.integer_variable import (
    DiscreteVariable,
)


class MarginalizationFactor(ContinuousInternalParameterlessFactor):
    def __init__(
        self,
        factor: ContinuousInternalParameterlessFactor,
        marginalized_variable: DiscreteVariable,
        conditioning_dict=None,
    ):
        if not isinstance(marginalized_variable, DiscreteVariable):
            raise ValueError("Only discrete variables can be marginalized.")

        variables = [var for var in factor.variables if var != marginalized_variable]
        super().__init__(variables, conditioning_dict)
        self.raw_factor = factor
        self.marginalized_variable = marginalized_variable

    def condition_on_non_empty_dict(self, assignment_dict):
        return MarginalizationFactor(
            self.raw_factor,
            self.marginalized_variable,
            self.total_conditioning_dict(assignment_dict),
        )

    def call_after_validation(self, assignment_dict, assignment_values):
        prob = 0.0
        for val in self.marginalized_variable.assignments:
            full_assignment_dict = {**assignment_dict, self.marginalized_variable: val}
            full_assignment_dict = self.total_conditioning_dict(full_assignment_dict)
            prob += self.raw_factor(full_assignment_dict)
        return prob
