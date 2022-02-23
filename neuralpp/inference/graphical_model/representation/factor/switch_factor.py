from neuralpp.inference.graphical_model.representation.factor.atomic_factor import AtomicFactor
from neuralpp.inference.graphical_model.representation.factor.continuous.marginalization_factor import \
    MarginalizationFactor
from neuralpp.util.util import join


class SwitchFactor(AtomicFactor):

    def __init__(self, switch, components):
        self.switch = switch
        self.components = components

    def call_after_validation(self, assignment_dict, assignment_values):
        switch_value = assignment_values[0]
        return self.components[switch_value](assignment_dict)

    def _transform_components(self, function):
        transformed_components = [function(c) for c in self.components]
        return SwitchFactor(self.switch, transformed_components)

    def condition_on_non_empty_dict(self, assignment_dict):
        if self.switch in assignment_dict:
            return self.components[assignment_dict[self.switch]].condition(assignment_dict)
        else:
            return self._transform_components(lambda c: c.condition(assignment_dict))

    def randomize(self):
        for c in self.components:
            c.randomize()

    def randomized_copy(self):
        return self._transform_components(lambda c: c.randomized_copy())

    def multiply_by_non_identity(self, other):
        return self._transform_components(lambda c: c.multiply_by_non_identity(other))

    def sum_out_variable(self, variable):
        if variable == self.switch:
            return MarginalizationFactor(self.switch, self)
        else:
            return self._transform_components(lambda c: c.sum_out_variable(variable))

    def __str__(self):
        return f"Switch factor based on {self.switch} with components {join(self.components)}"
