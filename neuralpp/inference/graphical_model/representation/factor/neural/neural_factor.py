import functools
from math import prod

import torch

from neuralpp.inference.graphical_model.representation.factor.atomic_factor import AtomicFactor
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import PyTorchTableFactor
from neuralpp.inference.graphical_model.representation.frame import dict_frame
from neuralpp.inference.graphical_model.representation.frame.dict_frame import convert_scalar_frame_to_tensor_frame, \
    convert_values_to_at_least_two_dimensions
from neuralpp.inference.graphical_model.representation.representation import contains_batch_coordinate
from neuralpp.inference.graphical_model.variable.discrete_variable import DiscreteVariable
from neuralpp.util import util
from neuralpp.util.util import join, find


def value_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    else:
        return torch.tensor([value], dtype=torch.float)


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

    def __init__(self, neural_net, input_variables, output_variable, conditioning_dict={}):

        all_variables = input_variables + [output_variable]

        super().__init__(self.non_conditioned_variables(input_variables, output_variable, conditioning_dict.keys()))

        assert isinstance(output_variable, DiscreteVariable), f"{NeuralFactor.__name__} requires output variable {output_variable} to be {DiscreteVariable.__name__}"
        assert output_variable.cardinality is not None, f"{NeuralFactor.__name__} requires output variable {output_variable} to have a well-defined cardinality, but {output_variable} has none"

        self.input_variables = input_variables
        self.free_input_variables = [v for v in input_variables if v not in conditioning_dict]
        self.output_variable = output_variable
        self.conditioning_dict = conditioning_dict
        self.neural_net = neural_net

    @staticmethod
    def non_conditioned_variables(input_variables, output_variable, conditioning_variables):
        return [v for v in input_variables + [output_variable] if v not in conditioning_variables]

    def pytorch_parameters(self):
        return self.neural_net.parameters()

    def call_after_validation(self, assignment_dict, assignment_values):
        probabilities = self.probabilities_from_assignment_dict(assignment_dict)
        output_value = assignment_dict[self.output_variable]
        if contains_batch_coordinate(assignment_values):
            # we used to have : for the first coordinate below, but that failed if output_value is a batch coordinate.
            # In that case, we would obtain the probability for each output value, per row.
            # Instead, we want the probability for each i-th output value from the *corresponding* i-th row.
            probability = probabilities[list(range(len(probabilities))), output_value]
        else:
            probability = probabilities[output_value]

        # # if output_value is a batch coordinate, probability will be a n x 1 tensor, but we
        # # want it to be a n-dimensional tensor.
        # if is_batch_coordinate(output_value):
        #     probability = probability.squeeze(1)

        return probability

    def probabilities_from_assignment_dict(self, assignment_dict):
        neural_net_input = self.neural_net_input_from_assignment_dict(assignment_dict)
        probabilities = self.output_probabilities(neural_net_input)
        return probabilities

    def neural_net_input_from_assignment_dict(self, assignment_dict):
        self.check_assignment_dict_is_complete(assignment_dict)
        self.check_assignment_dict_does_not_contradict_conditioning_dict(assignment_dict)
        assignment_and_conditioning_dict = util.union_of_dicts(assignment_dict, self.conditioning_dict)

        tuple_of_featurized_value_tensors = tuple(v.featurize(assignment_and_conditioning_dict[v])
                                                  for v in self.input_variables)
        try:
            neural_net_input = torch.cat(tuple_of_featurized_value_tensors)
        except Exception as e:
            raise Exception(f"Could not concatenate tensor values for {self.input_variables}") from e

        return neural_net_input

    def output_probabilities(self, neural_net_input):
        try:
            probabilities = self.neural_net(neural_net_input)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                raise Exception(f"Size mismatch exception when applying neural network {self.neural_net}, possibly due to mismatching feature vector (length {len(neural_net_input)}) and neural net input layer size. Original exception was: {e}")
            else:
                raise e
        # using shape[-1] takes care of both cases of probabilities being a batch or not.
        assert probabilities.shape[-1] == self.output_variable.cardinality, f"Neural net {self.neural_net} output must have the same size as output variable '{self.output_variable}' cardinality {self.output_variable.cardinality}, but has size {len(probabilities)} instead"
        return probabilities

    def condition_on_non_empty_dict(self, assignment_dict):
        # self.check_conditioning_is_on_factors_variables_only(assignment_dict)
        new_conditioning_dict = util.union_of_dicts(assignment_dict, self.conditioning_dict)
        return NeuralFactor(self.neural_net, self.input_variables, self.output_variable, new_conditioning_dict)

    def check_conditioning_is_on_factors_variables_only(self, assignment_dict):
        extra_variable = util.find(assignment_dict.keys(), lambda v: v not in self.variables)
        if extra_variable:
            raise Exception(
                f"Factor conditioned on {extra_variable} but that is not one of its variables {self.variables}. Factor is {self}")

    def randomize(self):
        self.neural_net.randomize()

    def randomized_copy(self):
        return \
            NeuralFactor(self.neural_net.randomized_copy(), self.input_variables, self.output_variable,
                         self.conditioning_dict)

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
        output_value = self.conditioning_dict[self.output_variable]

        def probability_of_completion_of_free_input_values(*free_input_values):
            total_assignment = {
                **self.conditioning_dict,
                **{var: val for var, val in zip(self.free_input_variables, free_input_values)},
                self.output_variable: output_value
            }
            return self(total_assignment)

        return PyTorchTableFactor.from_function(self.free_input_variables, probability_of_completion_of_free_input_values)

    def to_table_factor_if_output_variable_is_not_conditioned(self):
        relevant_conditioning_dict = {v: value for v, value in self.conditioning_dict.items()
                                      if v in self.input_variables}
        if len(relevant_conditioning_dict) > 0:
            broadcast_assignment_dict = dict_frame.broadcast_values(relevant_conditioning_dict)
            broadcast_assignment_dict = convert_scalar_frame_to_tensor_frame(broadcast_assignment_dict)
            broadcast_assignment_dict = convert_values_to_at_least_two_dimensions(broadcast_assignment_dict)
            conditioning_tensor = torch.cat(tuple(broadcast_assignment_dict.values()), dim=1)
        else:
            conditioning_tensor = torch.ones(1, 0)

        free_cardinalities = [torch.arange(free_variable.cardinality) for free_variable in self.free_input_variables]
        if len(free_cardinalities) > 0:
            assert len(conditioning_tensor.shape) == 2, \
                "Neural factor with unconditioned input variables must be defined on scalar-typed variables only"

            free_assignments = torch.cartesian_prod(*free_cardinalities)
            if len(free_cardinalities) == 1:
                free_assignments = free_assignments.unsqueeze(1)  # to make sure free_assignments is always 2D

            expanded_conditioning_tensor = conditioning_tensor.repeat_interleave(len(free_assignments), dim=0)

            expanded_free_assignments = free_assignments.repeat(len(conditioning_tensor), 1)
            expanded_free_assignments = expanded_free_assignments.to(expanded_conditioning_tensor.device).detach()

            all_inputs_tensor = torch.cat((expanded_conditioning_tensor, expanded_free_assignments), dim=1)
        else:
            all_inputs_tensor = conditioning_tensor

        probabilities = self.neural_net(all_inputs_tensor)

        number_of_batch_rows = probabilities.numel() // self.numel
        batch = number_of_batch_rows != 1
        if batch:
            probabilities_shape = (number_of_batch_rows, *self.non_batch_shape_including_output_variable)
        else:
            probabilities_shape = self.non_batch_shape_including_output_variable

        probabilities_in_right_shape = probabilities.reshape(probabilities_shape)

        resulting_factor = PyTorchTableFactor(self.free_input_variables + [self.output_variable],
                                              probabilities_in_right_shape, log_space=True, batch=batch)

        return resulting_factor

    def to_table_factor_if_output_variable_is_not_conditioned1(self):

        def get_completed_input_values_tensor(free_input_values):
            """
            Given a values assignment to free input variables
            return a tensor values assignment to all input variables
            by adding the conditioned input values to the values to free input variables.
            """

            def complete_with_conditioned_input_values(free_input_values):
                # TODO: need to review for cases where some values are batches and others are not
                index_of_next_free_input_value = 0
                input_values = []
                for v in self.input_variables:
                    if v in self.conditioning_dict:
                        next_value = self.conditioning_dict[v]
                    else:
                        next_value = free_input_values[index_of_next_free_input_value]
                        index_of_next_free_input_value += 1
                    input_values.append(next_value)
                return input_values

            def from_input_values_to_tensor(input_values):
                # TODO: simplify by making a list of tensors and stacking
                if len(input_values) == 1 and isinstance(input_values[0], torch.Tensor):  # optimization of singleton case
                    return input_values[0]
                else:
                    def horizontal_stack(t1, t2):
                        return torch.cat((t1, t2))
                    stacked_input_tensors = functools.reduce(horizontal_stack, map(value_tensor, input_values))
                    return stacked_input_tensors

            input_values = complete_with_conditioned_input_values(free_input_values)
            tensor = from_input_values_to_tensor(input_values)
            return tensor

        self._check_whether_free_input_variables_can_be_enumerated()
        assignments_of_free_input_variables = list(DiscreteVariable.assignments_product(self.free_input_variables))

        # TODO: should be creating a tensor directly without intermediary list:
        list_of_completed_input_values_tensors_for_each_free_input_variables_assignment = \
            list(map(get_completed_input_values_tensor, assignments_of_free_input_variables))
        input_tensor = torch.stack(list_of_completed_input_values_tensors_for_each_free_input_variables_assignment)

        probabilities_tensor = self.output_probabilities(input_tensor)

        number_of_batch_rows = probabilities_tensor.numel() // self.numel
        batch = number_of_batch_rows != 1
        if batch:
            probabilities_shape = (number_of_batch_rows, *self.non_batch_shape_including_output_variable)
        else:
            probabilities_shape = self.non_batch_shape_including_output_variable

        probabilities_tensor_in_right_shape = probabilities_tensor.reshape(probabilities_shape)

        resulting_factor = PyTorchTableFactor(self.free_input_variables + [self.output_variable],
                                              probabilities_tensor_in_right_shape, log_space=True, batch=batch)

        return resulting_factor

    @property
    @functools.lru_cache(1)
    def non_batch_shape_including_output_variable(self):
        return tuple([v.cardinality for v in self.free_input_variables] + [self.output_variable.cardinality])

    @property
    @functools.lru_cache(1)
    def numel(self):
        return prod(self.non_batch_shape_including_output_variable)

    def _check_whether_free_input_variables_can_be_enumerated(self):
        if any(v.assignments is None for v in self.free_input_variables):
            raise Exception(f"{type(self)}: input variables that are free at inference time "
                            f"must have method 'assignments'")

    def __repr__(self):
        result = repr(self.neural_net) + f" on {join(self.input_variables)} -> {self.output_variable}"
        if self.conditioning_dict:
            result += f" conditioned on {self.conditioning_dict}"
        return result

    def assignment_dict_does_not_contradict_conditioning(self, assignment_dict):
        for var, val in assignment_dict.items():
            if var in self.conditioning_dict and val != self.conditioning_dict[var]:
                return False
        return True

    def check_assignment_dict_is_complete(self, assignment_dict):
        assert all(v in assignment_dict for v in
                   self.variables), f"{NeuralFactor.__name__} received a non-complete assignment. It is missing the neural factor's variable '{find(self.variables, lambda v: v not in assignment_dict)}'. The assignment is {assignment_dict}"

    def check_assignment_dict_does_not_contradict_conditioning_dict(self, assignment_dict):
        assert self.assignment_dict_does_not_contradict_conditioning(
            assignment_dict), f"Assignment dict {assignment_dict} contradicts neural factor conditioning {self.conditioning_dict}"
