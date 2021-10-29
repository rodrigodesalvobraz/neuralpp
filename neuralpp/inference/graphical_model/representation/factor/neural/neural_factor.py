import functools
from math import prod

import torch
from neuralpp.inference.graphical_model.representation.factor.atomic_factor import (
    AtomicFactor,
)
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.representation.frame.dict_frame import (
    generalized_len_of_dict_frame, featurize_dict_frame, expand_univalues_in_dict_frame,
    concatenate_non_empty_dict_frame_into_single_2d_tensor, repeat_dict_frame, repeat_interleave_dict_frame,
    cartesian_product_of_tensor_dict_frames, concatenate_into_single_tensor, make_cartesian_features_dict_frame,
)
from neuralpp.inference.graphical_model.variable.discrete_variable import (
    DiscreteVariable,
)
from neuralpp.util import util
from neuralpp.util.util import find, join, is_iterable, expand_into_batch, cartesian_prod_2d, \
    cartesian_product_of_two_tensors, dict_slice


class NeuralFactor(AtomicFactor):
    """
    Factor based on a neural net, which must conform to the following interface:
    probability_tensor_indexed_by_output_variable_values = neural_net(features),
        for features the concatenation of feature vectors of all variables given full assignment
        and construction-time conditioning
    neural_net.randomize(): randomizes weights in-place
    neural_net.randomized_copy(): provides a copy of neural_net with randomized weights
    neural_net(concatenation of feature vectors of all variables, input ones and output one, given full assignment)

    For convenience, the last layer of the neural network may also just return
    the probabilities of the last n - 1 values of output_variable,
    in which case the 0-th probability will be computed as 1 - sum of the others.
    """

    def __init__(
            self, neural_net, input_variables, output_variable, conditioning_dict={}
    ):

        super().__init__(
            self.non_conditioned_variables(
                input_variables, output_variable, conditioning_dict.keys()
            )
        )

        assert isinstance(
            output_variable, DiscreteVariable
        ), f"{NeuralFactor.__name__} requires output variable {output_variable} to be {DiscreteVariable.__name__}"
        assert (
                output_variable.cardinality is not None
        ), f"{NeuralFactor.__name__} requires output variable {output_variable} to have a well-defined cardinality, " \
           f"but {output_variable} has none"

        self.input_variables = input_variables
        self.free_input_variables = [
            v for v in input_variables if v not in conditioning_dict
        ]
        self.output_variable = output_variable
        self.conditioning_dict = conditioning_dict
        self.neural_net = neural_net

    @staticmethod
    def non_conditioned_variables(
            input_variables, output_variable, conditioning_variables
    ):
        return [
            v
            for v in input_variables + [output_variable]
            if v not in conditioning_variables
        ]

    def pytorch_parameters(self):
        return self.neural_net.parameters()

    def call_after_validation_MODEL(self, assignment_dict):
        return self.to_table_factor_if_output_variable_is_conditioned_MODEL(assignment_dict).table.tensor
        # TODO: change underlying table from log to normal space

    def call_after_validation(self, assignment_dict, assignment_values):
        probabilities = self.probabilities_from_assignment_dict(assignment_dict)
        output_value = assignment_dict[self.output_variable]

        # the following code deals with the possibility that probabilities and output_value
        # differ in the length (number of values / batchness).
        # We could have normalized the assignment_dict to have all values of the same length,
        # but that might make the neural network run many times over the same input
        # (for example, if the input was univalue and the output multivalue,
        # then we would torch.expand the inputs and the neural network would apply
        # on many identical multiple roles).
        # So what we do instead is not to normalize the assignment, let probabilities
        # be computed in the most economical form given inputs,
        # and then deal with the possibility that probabilities and output_value
        # need to be adjusted to each other at this level.

        assert probabilities.dim() in {1, 2}
        assert isinstance(output_value, int) or is_iterable(output_value)

        if is_iterable(output_value):
            if probabilities.dim() == 1:
                probabilities = probabilities.expand(len(output_value), -1)
            if not isinstance(output_value, torch.Tensor):
                output_value = torch.tensor(output_value)
            probability = probabilities.gather(dim=1, index=output_value.unsqueeze(1)).squeeze()
        else:
            if probabilities.dim() == 1:
                probability = probabilities[output_value]
            else:
                probability = probabilities[:, output_value]

        return probability

    def probabilities_from_assignment_dict(self, assignment_dict):
        neural_net_input = self.neural_net_input_from_assignment_dict(assignment_dict)
        probabilities = self.output_probabilities(neural_net_input)
        return probabilities

    def neural_net_input_from_assignment_dict(self, assignment_dict):
        self.check_assignment_dict_is_complete(assignment_dict)
        self.check_assignment_dict_does_not_contradict_conditioning_dict(
            assignment_dict
        )
        assignment_and_conditioning_dict = util.union_of_dicts(
            assignment_dict, self.conditioning_dict
        )

        # featurize
        # check lengths
        # obtain multivalues length
        # if there are multivalues
        #     expand univalues into 2D tensors
        #     concatenate along dimension 1
        # else
        #     concatenate along dimension 0

        multivalue_lengths = set(v.multivalue_len(value)
                                 for v, value in assignment_and_conditioning_dict.items()
                                 if v.is_multivalue(value))

        if len(multivalue_lengths) > 1:
            raise Exception(f"neural factor received multivalue inputs of different lengths: {multivalue_lengths}")

        tuple_of_featurized_value_tensors = tuple(
            v.featurize(assignment_and_conditioning_dict[v])
            for v in self.input_variables
        )

        def variable_value_featurized():
            return (
                (v, assignment_and_conditioning_dict[v], fvt)
                for v, fvt in zip(self.input_variables, tuple_of_featurized_value_tensors)
            )

        there_are_multivalues = len(multivalue_lengths) == 1

        if there_are_multivalues:
            batch_size = next(iter(multivalue_lengths))
            tuple_of_featurized_value_tensors = tuple(
                expand_into_batch(fvt, batch_size) if not v.is_multivalue(value) else fvt
                for v, value, fvt in variable_value_featurized()
            )

        dim_to_concatenate = 1 if there_are_multivalues else 0
        neural_net_input = torch.cat(tuple_of_featurized_value_tensors, dim=dim_to_concatenate)

        return neural_net_input

    def output_probabilities(self, neural_net_input):
        try:
            probabilities = self.neural_net(neural_net_input)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                raise Exception(
                    f"Size mismatch exception when applying neural network {self.neural_net}, possibly due to mismatching feature vector (length {len(neural_net_input)}) and neural net input layer size. Original exception was: {e}"
                )
            else:
                raise e
        # using shape[-1] takes care of both cases of probabilities being a batch or not.
        assert (
                probabilities.shape[-1] == self.output_variable.cardinality
        ), f"Neural net {self.neural_net} output must have the same size as output variable '{self.output_variable}' cardinality {self.output_variable.cardinality}, but has size {len(probabilities)} instead"
        return probabilities

    def condition_on_non_empty_dict(self, assignment_dict):
        # self.check_conditioning_is_on_factors_variables_only(assignment_dict)
        new_conditioning_dict = util.union_of_dicts(
            assignment_dict, self.conditioning_dict
        )
        return NeuralFactor(
            self.neural_net,
            self.input_variables,
            self.output_variable,
            new_conditioning_dict,
        )

    def check_conditioning_is_on_factors_variables_only(self, assignment_dict):
        extra_variable = util.find(
            assignment_dict.keys(), lambda v: v not in self.variables
        )
        if extra_variable:
            raise Exception(
                f"Factor conditioned on {extra_variable} but that is not one of its variables {self.variables}. Factor is {self}"
            )

    def randomize(self):
        self.neural_net.randomize()

    def randomized_copy(self):
        return NeuralFactor(
            self.neural_net.randomized_copy(),
            self.input_variables,
            self.output_variable,
            self.conditioning_dict,
        )

    def mul_by_non_identity(self, other):
        return other * self.table_factor
        # order matters if other is also neural, because tables do not know how to multiply by neural factors.
        # This way, if other is also neural, it will itself be converted to a table.

    def sum_out_variable(self, variable):
        # TODO: This can be optimized for the case in which variable is the output variable (result is uniform factor
        # on free variables).
        return self.table_factor.sum_out_variable(variable)

    def argmax(self):
        return self.table_factor.argmax()

    def normalize(self):
        if self.variables == [self.output_variable]:
            return self
        else:
            return self.table_factor.normalize()

    def sample(self):
        return self.table_factor.sample()

    def single_sample(self):
        return self.table_factor.single_sample()

    @property
    @functools.lru_cache(1)
    def table_factor(self):
        if self.output_variable in self.conditioning_dict:
            return self.to_table_factor_if_output_variable_is_conditioned()
        else:
            return self.to_table_factor_if_output_variable_is_not_conditioned()

    def to_table_factor_if_output_variable_is_conditioned(self):
        table_factor = self.to_table_factor_if_output_variable_is_not_conditioned()
        return table_factor.condition({self.output_variable: self.conditioning_dict[self.output_variable]})

    def to_table_factor_if_output_variable_is_not_conditioned(self):
        probabilities_tensor = self.compute_probability_tensor()
        resulting_factor = self.make_table_factor_for_free_and_output_variables(probabilities_tensor)
        return resulting_factor

    def compute_probability_tensor(self):
        return self.compute_probabilities_tensor_for(self.conditioning_dict)

    def compute_probabilities_tensor_for(self, assignment_dict, no_free_variables = False):
        """
        Returns the probabilities tensor produced by the neural net for a given assignment_dict.
        The probabilities tensor will have an optional batch dimension (determined by whether any
        values in the conditioning dict is multivalue OR if there are free input variables that will be automatically
        filled in with all assignments), followed by other dimensions in the order of self.input_variables.

        If the caller knows there are no free variables, they can pass no_free_variables = True for optimization.
        In this case, the assignment_dict keys are required to be in the same order as self.input_variables.

        If there are free variables, they are assumed to be IntegerVariables.
        """
        relevant_conditioning_dict_frame = dict_slice(assignment_dict, self.input_variables)
        featurized_conditioning_dict_frame = featurize_dict_frame(relevant_conditioning_dict_frame)
        expanded_featurized_conditioning_dict_frame = expand_univalues_in_dict_frame(featurized_conditioning_dict_frame)

        if no_free_variables:
            ordered_inputs_tensor = expanded_featurized_conditioning_dict_frame
        else:
            ordered_inputs_tensor = self.complete_with_free_variables(expanded_featurized_conditioning_dict_frame)

        ordered_inputs_tensor = concatenate_into_single_tensor(ordered_inputs_tensor)
        probabilities_tensor = self.neural_net(ordered_inputs_tensor)
        return probabilities_tensor

    def complete_with_free_variables(self, expanded_featurized_conditioning_dict_frame):
        free_input_variables = self.input_variables - expanded_featurized_conditioning_dict_frame.keys()
        cartesian_free_features_dict_frame = make_cartesian_features_dict_frame(free_input_variables)
        all_inputs_dict_frame = cartesian_product_of_tensor_dict_frames(cartesian_free_features_dict_frame,
                                                                        expanded_featurized_conditioning_dict_frame)
        ordered_inputs_tensor = {variable: all_inputs_dict_frame[variable] for variable in self.input_variables}
        return ordered_inputs_tensor

    def make_cartesian_features_dict_frame(variables):
        if len(variables) > 0:
            free_cardinalities = [torch.arange(fv.cardinality) for fv in variables]
            free_assignments = cartesian_prod_2d(free_cardinalities)
            cartesian_free_features_dict_frame = {variable: free_assignments[:, i]
                                                  for i, variable in enumerate(variables)}
        else:
            cartesian_free_features_dict_frame = {}
        return cartesian_free_features_dict_frame

    def make_table_factor_for_free_and_output_variables(self, probabilities_tensor):
        number_of_batch_rows = probabilities_tensor.numel() // self.number_of_probabilities_per_batch_row
        batch = number_of_batch_rows != 1
        if batch:
            probabilities_shape = (
                number_of_batch_rows,
                *self.non_batch_shape_including_output_variable,
            )
        else:
            probabilities_shape = self.non_batch_shape_including_output_variable
        probabilities_tensor_in_right_shape = probabilities_tensor.reshape(probabilities_shape)
        resulting_factor = PyTorchTableFactor(
            self.free_input_variables + [self.output_variable],
            probabilities_tensor_in_right_shape,
            log_space=True,
            batch=batch,
        )
        return resulting_factor

    @property
    @functools.lru_cache(1)
    def non_batch_shape_including_output_variable(self):
        return tuple(v.cardinality for v in self.free_input_variables) + (self.output_variable.cardinality,)

    @property
    @functools.lru_cache(1)
    def number_of_probabilities_per_batch_row(self):
        return prod(self.non_batch_shape_including_output_variable)

    def _check_whether_free_input_variables_can_be_enumerated(self):
        if any(v.assignments is None for v in self.free_input_variables):
            raise Exception(
                f"{type(self)}: input variables that are free at inference time "
                f"must have method 'assignments'"
            )

    def __repr__(self):
        result = (
                repr(self.neural_net)
                + f" on {join(self.input_variables)} -> {self.output_variable}"
        )
        if self.conditioning_dict:
            result += f" conditioned on {self.conditioning_dict}"
        return result

    def assignment_dict_does_not_contradict_conditioning(self, assignment_dict):
        for var, val in assignment_dict.items():
            if var in self.conditioning_dict and val != self.conditioning_dict[var]:
                return False
        return True

    def check_assignment_dict_is_complete(self, assignment_dict):
        assert all(
            v in assignment_dict for v in self.variables
        ), f"{NeuralFactor.__name__} received a non-complete assignment. It is missing the neural factor's variable '{find(self.variables, lambda v: v not in assignment_dict)}'. The assignment is {assignment_dict}"

    def check_assignment_dict_does_not_contradict_conditioning_dict(
            self, assignment_dict
    ):
        assert self.assignment_dict_does_not_contradict_conditioning(
            assignment_dict
        ), f"Assignment dict {assignment_dict} contradicts neural factor conditioning {self.conditioning_dict}"
