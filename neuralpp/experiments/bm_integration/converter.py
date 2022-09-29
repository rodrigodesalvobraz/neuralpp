from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
)
from neuralpp.inference.graphical_model.variable.variable import Variable
import beanmachine.ppl as bm
from typing import (
    Callable,
    List,
    Optional,
    overload,
    Union,
    Dict,
    Collection,
    Mapping,
    TypeVar,
)
from neuralpp.experiments.bm_integration.distributions import get_distribution
import torch
from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    ProductFactor,
)

# a generic typevar to match all subclasses of Variable
V = TypeVar("V", bound=Variable)


class BeanMachineConverter:
    def __init__(
        self,
        factors: Optional[Collection[Factor]] = None,
        assignment_dict: Optional[Mapping[V, torch.Tensor]] = None,
    ):
        # a mapping from variable to the corresponding @random_variable function
        self._from_variable_to_rv_function: Dict[Variable, Callable] = {}
        self._observations: Dict[bm.RVIdentifier, torch.Tensor] = {}

        if factors is not None:
            self._register_factors(factors)
        if assignment_dict is not None:
            self.register_observations(assignment_dict)

    @property
    def observations(self) -> Dict[bm.RVIdentifier, torch.Tensor]:
        return self._observations

    def _register_factors(self, factors: Collection[Factor]) -> None:
        """
        Convert each of the factors to a @random_variable and register it in self.
        """
        for f in factors:
            self._register_factor(f)

    def register_observations(
        self, assignment_dict: Mapping[V, torch.Tensor]
    ) -> Dict[bm.RVIdentifier, torch.Tensor]:
        """
        Convert an assignment_dict into an RVIdentifier to torch.Tensor mapping that
        can be pass to Bean Machine's infer method. This method should be invoked
        after all factors have been registered.
        """
        for variable, value in assignment_dict.items():
            if variable in self._from_variable_to_rv_function:
                rv_id = self.invoke_rv_function_of(variable)
                self._observations[rv_id] = value
            else:
                self._register_functional(variable, value)
        return self.observations

    def invoke_rv_function_of(self, variable: Variable):
        """
        Invoke the @random_variable function corresponds to the given variable.
        During inference, this will return the value of a random variable. If the
        method is called outside of an inference scope, this will return an
        RVIdentifier instead.
        """
        assert (
            variable in self._from_variable_to_rv_function
        ), f"Could not find definition for variable '{variable}'"
        return self._from_variable_to_rv_function[variable]()

    def _register_factor(self, factor: Factor) -> None:
        """Convert a given factor to a @random_variable and register it to self"""
        if isinstance(factor, ProductFactor):
            return self._register_factors(ProductFactor.factors(factor))

        parent_variables = factor.variables[1:]
        child_variable = factor.variables[0]

        @bm.random_variable
        def rv_function():
            parent_values = {p: self.invoke_rv_function_of(p) for p in parent_variables}
            factor_on_child = factor.condition(parent_values)
            return get_distribution(child_variable, factor_on_child)

        # assign local function a name to distinguish them
        rv_function.__wrapped__.__name__ = child_variable.name
        self._from_variable_to_rv_function[child_variable] = rv_function

    def _register_functional(self, variable: Variable, value: torch.Tensor):
        """
        Convert a variable without a prior distribution to a deterministic
        @functional
        """

        @bm.functional
        def rv_function():
            return value

        rv_function.__wrapped__.__name__ = variable.name
        self._from_variable_to_rv_function[variable] = rv_function
