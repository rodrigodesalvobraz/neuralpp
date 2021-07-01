class Variable:
    def __eq__(self, other):
        self._not_implemented("__eq__")

    def __hash__(self):
        self._not_implemented("__hash__")

    def __repr__(self):
        self._not_implemented("__repr__")

    def _not_implemented(self, name):
        error = NotImplementedError(f"{name} not implemented for {type(self)}")
        raise error
