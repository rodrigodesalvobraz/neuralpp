class ConditionalProbability(AtomicFactor):

    @property
    def children(self):
        util.not_implemented(self, "children")

    @children.setter
    def children(self, new_value):
        util.not_implemented(self, "children.setter")

    @property
    def parents(self):
        util.not_implemented(self, "parents")

    @children.setter
    def parents(self, new_value):
        util.not_implemented(self, "parents.setter")
