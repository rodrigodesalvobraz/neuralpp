import torch


class TensorMixedRadix:
    """
    Converter from integers to their mixed radix representation.
    """

    def __init__(self, radices):
        self.strides, self.max_value = self.compute_strides_and_max_value(radices)

    def representation(self, values: torch.Tensor):
        """
        Given a tensor of integral values in the last dimension, returns
        a tensor where each integral value is replaced by a 1-dimensional tensor
        containing its mixed radix representation with given radices.
        As a consequence, the shape of the result will be (*values.shape, len(self.strides))
        """
        if values.numel() == 0:
            return torch.zeros((*values.shape, len(self.strides)), dtype=torch.int)

        if (m := values.max()) > self.max_value:
            raise MaxValueException(m, self.max_value)

        digits = []
        for i, stride in enumerate(self.strides):
            i_th_digits = (values // stride).int()
            values -= i_th_digits * stride
            digits.append(i_th_digits)

        digits_as_vectors = [torch.unsqueeze(i_th_digits, dim=-1) for i_th_digits in digits]
        representation = torch.cat(digits_as_vectors, dim=-1)

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
