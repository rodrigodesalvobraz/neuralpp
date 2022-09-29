from neuralpp.inference.graphical_model.representation.factor.atomic_factor import (
    AtomicFactor,
)
from neuralpp.inference.graphical_model.representation.factor.factor import (
    Factor,
)
from neuralpp.inference.graphical_model.representation.table.table_util import (
    index_of,
    permutation_from_to,
    shape,
)
from neuralpp.inference.graphical_model.variable.discrete_variable import (
    DiscreteVariable,
)
from neuralpp.util.discrete_sampling import discrete_sample
from neuralpp.util.util import join


class TableFactor(AtomicFactor):
    """
    TableFactor is a Factor based on a Table.
    Like Table, a TableFactor may be a batch or not.
    """

    def __init__(self, variables, table):
        super().__init__(variables)
        self.table = table

    def new_instance(self, variables, table):
        return type(self)(variables, table)

    @property
    def batch(self):
        return self.table.batch

    def assignments(self):
        return DiscreteVariable.assignments_product(self.variables)

    def condition_on_non_empty_dict(self, assignment_dict):
        non_batch_slice_coordinates = tuple(
            assignment_dict[v] if v in assignment_dict else slice(None)
            for v in self.variables
        )
        non_conditioned_variables = [
            v
            for v in self.variables
            if v not in assignment_dict
            or isinstance(assignment_dict[v], slice)
        ]
        conditioned_table = self.table.slice(non_batch_slice_coordinates)
        return self.new_instance(non_conditioned_variables, conditioned_table)

    def call_after_validation(self, assignment_dict, assignment_values):
        return self.table[assignment_values]

    def mul_by_non_identity(self, other):
        """Multiplies factors so that (f1 * f2)(assignment) = f1(assignment)*f2(assignment)"""

        # TableFactor only knows how to multiply by another TableFactor.
        # If other is not TableFactor, we ask that it handle the multiplication.
        if not isinstance(other, TableFactor):
            return other * self

        """
        Implementation rationale:
        Let 'common' be the shared variables between the two factors, and e1 and e2
        the variables exclusive to each representation respectively.

        We want to compute f(e1, e2, common) = f1(e1, common) * f2(e2, common)
        Such multiplication operation is not readily available, however.
        Instead, we use component-wise multiplication by computing:
        for all e1, e2, common:
        f1'(e1, e2, common) = f1(e1, common)
        f2'(e1, e2, common) = f2(e2, common)
        f(e1, e2, common) = f1'(e1, e2, common) * f2'(e1, e2, common)

        We compute f1' and f2' by using Table.expand

        Note that things work transparently if tables are batches.
        """

        common = list(set(self.variables) & set(other.variables))

        (
            f1_e1_common_table,
            e1,
        ) = self.get_permuted_table_with_selected_variables_moved_to_the_end(
            self, common
        )
        (
            f2_e2_common_table,
            e2,
        ) = self.get_permuted_table_with_selected_variables_moved_to_the_end(
            other, common
        )

        result_variables = e1 + e2 + common

        f1p_table = f1_e1_common_table.expand(
            shape_to_be_inserted=shape(e2), dim=len(e1)
        )
        f2p_table = f2_e2_common_table.expand(
            shape_to_be_inserted=shape(e1), dim=0
        )
        result_table = f1p_table * f2p_table

        result = self.new_instance(result_variables, result_table)

        return result

    @staticmethod
    def get_permuted_table_with_selected_variables_moved_to_the_end(
        factor, selected_variables
    ):
        other_variables = [
            v for v in factor.variables if v not in selected_variables
        ]
        variables_in_desired_order = other_variables + selected_variables
        permuted_table = factor.table.permute(
            index_of(variables_in_desired_order, factor.variables)
        )
        return permuted_table, other_variables

    def sum_out_variable(self, variable):
        result_variables = [v for v in self.variables if v != variable]
        index_of_variable = self.variables.index(variable)
        result_table = self.table.sum_out(index_of_variable)
        result = self.new_instance(result_variables, result_table)
        return result

    def argmax(self):
        indices = self.table.argmax()
        if (
            len(self.variables) == 1
        ):  # if there is a single variable, indices is 1D
            assignment_getter = (
                lambda var_index: indices
            )  # regardless of being batch or not
        else:
            assignment_getter = (
                lambda var_index: indices[:, var_index]
                if self.batch
                else indices[var_index]
            )
        assignment_dict = {
            v: assignment_getter(var_index)
            for var_index, v in enumerate(self.variables)
        }
        return assignment_dict

    def normalize(self):
        return self.new_instance(self.variables, self.table.normalize())

    def randomize(self):
        self.table.randomize()

    def randomized_copy(self):
        return self.new_instance(self.variables, self.table.randomized_copy())

    def sample(self, n=1):
        return self.table.sample(n)

    def single_sample(self):
        if len(self.table) == 0:
            return []
        else:
            entries_sum = self.table.sum()
            if abs(entries_sum - 1.0) > 0.00001:
                raise Exception(
                    f"Sampled factor is not normalized. Sum of entries is {entries_sum}"
                )
            return discrete_sample(
                self.assignments(), lambda assignment: self.table[assignment]
            )

    def __eq__(self, other):
        """
        Compares factors by checking if they have the same variables and if tables are equal according to ==
        (after appropriate permutation if variables are not in the same order).
        """
        if isinstance(other, TableFactor):
            if self.variables == other.variables:
                return self.table == other.table
            elif set(self.variables) == set(other.variables):
                return (
                    self.table
                    == other.get_table_permuted_to_agree_with_table_of(self)
                )
            else:
                return False
        elif not isinstance(other, Factor):
            return False
        else:
            raise Exception(
                f"Comparison of TableFactor to factors other than TableFactor is not implemented. "
                f"Got {type(other)}"
            )

    def get_table_permuted_to_agree_with_table_of(self, other):
        return self.table.permute(
            permutation_from_to(self.variables, other.variables)
        )

    @property
    def table_factor(self):
        return self

    def __repr__(self):
        return "Factor on (" + join(self.variables) + "): " + repr(self.table)

    def __str__(self):
        return "Factor on (" + join(self.variables) + "): " + str(self.table)
