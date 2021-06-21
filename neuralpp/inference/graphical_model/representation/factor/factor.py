import functools

from neuralpp.util.group import Group
import collections


class Factor:

    def __init__(self, variables):
        self.variables = variables

    def __contains__(self, variable):
        return variable in self.variables

    def __call__(self, assignment_dict):
        assignment_values = self.validate_argument_to_call(assignment_dict)
        return self.call_after_validation(assignment_dict, assignment_values)

    def call_after_validation(self, assignment_dict, assignment_values):
        """
        Receives the original assignment_dict passed to __call__ after validation as
        well as a list of values, which is a sub-product of the validation,
        in the same order as their respective variables in this factor.
        Implementations can make use of either one to compute the potential.
        """
        self._not_implemented("call_after_validation")

    def __getitem__(self, assignment_dict):
        return self(assignment_dict)

    def condition(self, assignment_dict):
        """
        Receives an assignment dict from variables to values and returns the result of conditioning the factor
        by that assignment.
        This default implementation checks if the assignment dict is indeed dict-like
        (instance of collections.abc.Mapping).
        If it is, it also checks whether the assignment dict is empty and, if so, the factor is returned unchanged.
        Otherwise, this default implementation delegates to condition_on_non_empty_dict(assignment_dict)
        for class-specific processing.
        """
        if isinstance(assignment_dict, collections.abc.Mapping):
            if assignment_dict:
                return self.condition_on_non_empty_dict(assignment_dict)
            else:
                return self
        else:
            raise Exception(f"Factor being conditioned on non-dict-like object {assignment_dict}")

    def condition_on_non_empty_dict(self, assignment_dict):
        self._not_implemented("condition_on_non_empty_dict")

    def randomize(self):
        """Randomized parameters in-place"""
        self._not_implemented("randomized")

    def randomize_copy(self):
        """
        Returns a representation with same structure but randomized parameters
        IMPORTANT: this will create new copies of internal structures, de-coupling parameters
        shared with other factors.
        """
        self._not_implemented("randomized_copy")

    def pytorch_parameters(self):
        self._not_implemented("pytorch_parameters")

    def __mul__(self, other):
        if other is Group.identity:
            return self
        else:
            return self.mul_by_non_identity(other)

    __rmul__ = __mul__

    def mul_by_non_identity(self, other):
        self._not_implemented("mul_by_non_identity")

    def __xor__(self, variable_or_variables):
        if type(variable_or_variables) in [set, list, tuple]:
            return self.sum_out_variables(variable_or_variables)
        else:
            return self.sum_out_variable(variable_or_variables)

    def sum_out_variables(self, variables):
        current = self
        for v in variables:
            current = current ^ v
        return current

    def sum_out_variable(self, variable):
        self._not_implemented("sum_out_variable")

    def argmax(self):
        """
        Returns an assignment dictionary for the largest value in the factor.
        If the factor is a batch, the assignment dict will also have batch values.
        """
        self._not_implemented("argmax")

    def normalize(self):
        self._not_implemented("normalize")

    def atomic_factor(self):
        self._not_implemented("atomic_factor")

    def _not_implemented(self, name):
        # creating a variable first prevents compiler from thinking this is an abstract method
        error = NotImplementedError(f"{name} not implemented for {type(self)}")
        raise error

    @property
    @functools.lru_cache(1)
    def table_factor(self):
        """Table factor equivalent to self"""
        self._not_implemented("to")

    # Convenience methods

    def from_assignment_to_assignment_dict(self, assignment):
        return dict(zip(self.variables, assignment))

    def validate_argument_to_call(self, assignment_dict):
        try:
            assignment_values = [assignment_dict[v] for v in self.variables]
        except TypeError as e:
            if isinstance(assignment_dict, (list, tuple)):
                raise Exception(
                    f"Factors must be called on an assignment dict from variables to values, not on a {type(assignment_dict).__name__}: {assignment_dict}") \
                    from e
            else:
                raise Exception(f"Type error computing the application of a {type(self).__name__} to an assignment, "
                                f"which must be a dict. The argument was " f"{assignment_dict}") \
                    from e
        except KeyError as e:
            raise Exception(f"Indexed access to Factor requires a complete assignment to its variables {self.variables}, but got {assignment_dict} which does not include a variable: {e}")

        return assignment_values
