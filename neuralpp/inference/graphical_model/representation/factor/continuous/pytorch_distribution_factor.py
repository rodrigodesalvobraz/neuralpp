from neuralpp.inference.graphical_model.representation.factor.continuous.continuous_internal_parameterless_factor import (
    ContinuousInternalParameterlessFactor,
)


class PyTorchDistributionFactor(ContinuousInternalParameterlessFactor):
    """
    A specialization of ContinuousInternalParameterlessFactor based on a PyTorch distribution.
    Note that even though PyTorch distributions do have parameters,
    here they are not internal, but rather provided by *external* random variables in the model.
    """

    def __init__(
        self,
        pytorch_distribution_maker,
        all_variables_including_conditioned_ones,
        conditioning_dict=None,
    ):
        """
        Makes a PyTorchDistributionFactor based on given PyTorch distribution.

        pytorch_distribution_maker receives an assignment to the parameters to a PyTorch distribution
        and returns a PyTorch distribution.

        The first element of 'all_variables_including_conditioned_ones' must be the value following the distribution,
        and the remaining ones must be the distribution parameters in the same
        order used by the distribution maker function.

        Note that self.variables will contain only the non-conditioned variables.
        """
        conditioning_dict = conditioning_dict or {}
        super().__init__(
            [
                v
                for v in all_variables_including_conditioned_ones
                if v not in conditioning_dict
            ],
            conditioning_dict,
        )
        self.pytorch_distribution_maker = pytorch_distribution_maker
        self.all_variables_including_conditioned_ones = (
            all_variables_including_conditioned_ones
        )
        self.value_variable = all_variables_including_conditioned_ones[0]
        self.distribution_parameter_variables = (
            all_variables_including_conditioned_ones[1:]
        )
        self.unconditioned_parameters = [
            v for v in self.variables[1:] if v not in self.conditioning_dict
        ]

    def condition_on_non_empty_dict(self, assignment_dict):
        return type(self)(
            self.all_variables_including_conditioned_ones,
            self.total_conditioning_dict(assignment_dict),
        )

    def call_after_validation(self, assignment_dict, assignment_values):
        assignment_and_conditioning_dict = self.total_conditioning_dict(
            assignment_dict
        )
        distribution_parameters = [
            assignment_and_conditioning_dict[variable]
            for variable in self.distribution_parameter_variables
        ]
        distribution = self.pytorch_distribution_maker(
            *distribution_parameters
        )
        value = assignment_and_conditioning_dict[self.value_variable]
        return distribution.log_prob(value).exp()

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (
            f"{self.variables[0]} ~ {type(self).__name__}("
            + str(self.unconditioned_parameters)
            + ", "
            + (str(self.conditioning_dict) if self.conditioning_dict else "")
            + ")"
        )
