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
        self._rv_functions: Dict[Variable, Callable] = {}
        self._observations: Dict[bm.RVIdentifier, torch.Tensor] = {}

        if factors is not None:
            self.register_factors(factors)
        if assignment_dict is not None:
            self.register_observations(assignment_dict)

    @property
    def observations(self) -> Dict[bm.RVIdentifier, torch.Tensor]:
        return self._observations

    def register_factors(self, factors: Collection[Factor]) -> None:
        """Convert each of the fector to a @random_variable and add register it in
        self"""
        for f in factors:
            self._make_random_variable(f)

    def register_observations(
        self, assignment_dict: Mapping[V, torch.Tensor]
    ) -> Dict[bm.RVIdentifier, torch.Tensor]:
        """Convert an assignment_dict into an RVIdentifier to torch.Tensor mapping that
        can be pass to Bean Machine's infer method. This method should be invoked
        after all factors have been registered."""
        for variable, obs_value in assignment_dict.items():
            if variable in self._rv_functions:
                rvid = self.invoke(variable)
                self._observations[rvid] = obs_value
            else:
                self._make_functional(variable, obs_value)
        return self.observations

    def invoke(self, variable: Variable):
        """Invoke the @random_variable function corresponds to the given variable.
        During inference, this will return the value of a random variable. If the
        method is called outside of an inference scope, this will return an
        RVIdentifier instead."""
        return self._rv_functions[variable]()

    def _make_random_variable(self, factor: Factor) -> None:
        """Convert a given factor to a @random_variable and register it to self"""
        if isinstance(factor, ProductFactor):
            return self.register_factors(ProductFactor.factors(factor))

        parent_vars = factor.variables[1:]
        child_var = factor.variables[0]

        @bm.random_variable
        def rvfunction():
            parent_values = {p: self.invoke(p) for p in parent_vars}
            factor_on_child = factor.condition(parent_values)
            return get_distribution(child_var, factor_on_child)

        self._rv_functions[child_var] = rvfunction

    def _make_functional(self, variable: Variable, value: torch.Tensor):
        """Convert a variable without a prior distribution to a deterministic
        @functional"""

        @bm.functional
        def rvfunction():
            return value

        self._rv_functions[variable] = rvfunction
