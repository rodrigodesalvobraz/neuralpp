class AbstractConditionalProbability(ConditionalProbability):
    def __init__(self, parents, children, edges):
        self.parents = parents
        self.children = children
        self.edges = edges

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

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def parents(self, new_value):
        self._edges = new_value
        return self._edges
