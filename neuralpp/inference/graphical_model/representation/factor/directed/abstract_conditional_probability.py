class AbstractConditionalProbability(AtomicFactor):

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, new_value):
        self._children = new_value
        return self._children

    @property
    def parents(self):
        return self._parents

    @children.setter
    def parents(self, new_value):
        self._parents = new_value
        return self._parents
