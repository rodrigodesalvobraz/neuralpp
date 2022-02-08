class Edge:
    """
    An edge from a parent variable to a child variable.

    'path' is a tuple representing one of the paths (arbitrarily chosen) that gave rise to
    this edge through the process of variable elimination (when a variable is eliminated,
    their neighbors are connected as new edges).

    If a path is not provided (meaning this edge did not originated from a variable elimination),
    then this should be (parent, child).
    """

    def __init__(self, parent, child, path=None):
        self.parent = parent
        self.child = child
        if path is None:
            path = (parent, child)
        self.path = path

    def members(self):
        return (self.parent, self.child, self.path)

    def __eq__(self, other):
        return isinstance(other, Edge) and self.members() == other.members()

    def __hash__(self):
        m = self.members()
        return hash(m)

    def __repr__(self):
        return str(self.members())

    def __iter__(self):
        yield self.parent
        yield self.child
