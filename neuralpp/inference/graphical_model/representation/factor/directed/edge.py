class Edge:

    @property
    def source(self):
        util.not_implemented(self, "source")

    @source.setter
    def source(self, new_value):
        util.not_implemented(self, "source")

    @property
    def destination(self):
        util.not_implemented(self, "destination.setter")

    @destination.setter
    def destination(self, new_value):
        util.not_implemented(self, "destination.setter")

    @property
    def paths(self):
        """
        A generator of paths that gave rise to this edge through the process of variable elimination.
        As a variable are eliminated, their neighbors are connected as new edges.
        """
        util.not_implemented(self, "path")

    @paths.setter
    def paths(self, new_value):
        util.not_implemented(self, "path.setter")

