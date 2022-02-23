import torch


class TensorMixedRadix:
    """
    Converter from integers to their mixed radix representation.
    """

    def __init__(self, radices):
        self.strides, self.max_value = self.compute_strides_and_max_value(radices)

    def representation(self, values: torch.Tensor):
        """
        Given a 1-dimensional tensor of values, returns a 2-dimensional tensor
        where each row i is a tensor with the digits of the representation of values[i]
        in a mixed radix representation with given radices.
        """

        number_of_values = values.shape[0]

        if number_of_values == 0:
            return torch.zeros(0, len(self.strides))

        if (m := max(values)) > self.max_value:
            raise MaxValueException(m, self.max_value)

        representation = torch.zeros((number_of_values, 0), dtype=torch.int)
        for i in range(len(self.strides)):
            i_th_digits = torch.tensor(values // self.strides[i], dtype=torch.int)
            representation = torch.cat(
                (representation, i_th_digits.reshape(number_of_values, 1)), dim=1
            )
            values -= i_th_digits * self.strides[i]
        return representation

    @staticmethod
    def compute_strides_and_max_value(radices):
        if len(radices) == 0:
            return [], 0
        strides = [1] * len(radices)
        for i in range(len(radices) - 2, -1, -1):
            # i in len(radices) - 2, ..., 0
            strides[i] = radices[i + 1] * strides[i + 1]
        max_value = radices[0] * strides[0] - 1
        return strides, max_value


class MaxValueException(Exception):
    def __init__(self, violating_value, max_value):
        super().__init__(
            f"Received value {violating_value} greater than mixed radix max value {max_value}"
        )
        self.violating_value = violating_value
        self.max_value = max_value

    def __eq__(self, other):
        return (
            isinstance(other, MaxValueException)
            and self.violating_value == other.violating_value
            and self.max_value == other.max_value
        )
