from neuralpp.inference.graphical_model.representation.factor.continuous.continuous_internal_parameterless_factor import (
    ContinuousInternalParameterlessFactor,
)


class PyTorchDistributionFactor(ContinuousInternalParameterlessFactor):
    """
    A specialization of ContinuousInternalParameterlessFactor based on a PyTorch distribution.
    """

    def __init__(self, pytorch_distribution_maker, variables, conditioning_dict=None):
        """
        Makes a PyTorchDistributionFactor based on given PyTorch distribution.
        The first element of 'variables' must be the value following the distribution,
        and the remaining ones must be the distribution parameters in the same
        order used by the distribution maker function.
        """
        conditioning_dict = conditioning_dict or {}
        super().__init__(
            [v for v in variables if v not in conditioning_dict], conditioning_dict
        )
        self.pytorch_distribution_maker = pytorch_distribution_maker
        self.value_variable = self.variables[0]
        self.distribution_parameter_variables = variables[1:]

    def condition_on_non_empty_dict(self, assignment_dict):
        return type(self)(self.variables, self.total_conditioning_dict(assignment_dict))

    def call_after_validation(self, assignment_dict, assignment_values):
        assignment_and_conditioning_dict = self.total_conditioning_dict(assignment_dict)
        distribution_parameters = [
            assignment_and_conditioning_dict[variable]
            for variable in self.distribution_parameter_variables
        ]
        distribution = self.pytorch_distribution_maker(*distribution_parameters)
        value = assignment_and_conditioning_dict[self.value_variable]
        return distribution.log_prob(value).exp()
