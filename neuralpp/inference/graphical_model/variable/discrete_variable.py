import itertools

from neuralpp.inference.graphical_model.variable.variable import Variable


class DiscreteVariable(Variable):
    @staticmethod
    def assignments_product(variables):
        if len(variables) != 0:
            return itertools.product(*[v.assignments() for v in variables])
        else:
            return [tuple()]

    @staticmethod
    def assignments_product_dicts(variables):
        def make_dict(values):
            return {var: val for var, val in zip(variables, values)}

        return map(make_dict, DiscreteVariable.assignments_product(variables))

    def __init__(self, name=None, cardinality=None):
        super().__init__()
        self._name = name
        self._cardinality = cardinality

    @property
    def cardinality(self):
        return self._cardinality

    @cardinality.setter
    def cardinality(self, new_value):
        self._cardinality = new_value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_value):
        self._name = new_value

    def assignments(self):
        self._not_implemented("assignments")

    def featurize(self, value):
        self._not_implemented("featurize")

    def _not_implemented(self, name):
        error = NotImplementedError(f"{name} not implemented for {type(self)}")
        raise error
