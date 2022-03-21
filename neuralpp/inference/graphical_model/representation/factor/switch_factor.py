from typing import List

from neuralpp.inference.graphical_model.representation.factor.atomic_factor import AtomicFactor
from neuralpp.inference.graphical_model.representation.factor.continuous.mixture_factor import \
    MixtureFactor
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util import util
from neuralpp.util.util import join


class SwitchFactor(AtomicFactor):
    def __init__(self, switch, components):
        variables: List[Variable] = util.ordered_union_list(component.variables for component in components)
        variables.append(switch)
        variables = list(variables)
        super().__init__(variables)
        self.switch = switch
        self.components = components

    def call_after_validation(self, assignment_dict, assignment_values):
        switch_value = assignment_dict[self.switch]
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

    def mul_by_non_identity(self, other):
        return self._transform_components(lambda c: c.mul_by_non_identity(other))

    def sum_out_variable(self, variable):
        if variable == self.switch:
            return MixtureFactor(self.switch, self)
        else:
            return self._transform_components(lambda c: c.sum_out_variable(variable))

    def __eq__(self, other):
        if not isinstance(other, SwitchFactor):
            return False
        return self.switch == other.switch and self.components == other.components

    def __hash__(self):
        return hash(self.switch) + hash(self.components)

    def __repr__(self):
        return f"Switch({repr(self.switch)}, {repr(self.components)})"

    def __str__(self):
        return f"Switch factor based on {self.switch} with components {join(self.components)}"
