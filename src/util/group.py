import operator
from functools import reduce


class Group:

    identity = object()

    @staticmethod
    def product(elements):
        return reduce(operator.mul, elements, Group.identity)


