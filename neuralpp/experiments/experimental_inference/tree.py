from functools import reduce

from neuralpp.inference.graphical_model.representation.factor.product_factor import Factor
from neuralpp.inference.graphical_model.variable.variable import Variable


class Graph:
    def neighbors(self, node):
        raise NotImplemented()


class FactorGraph(Graph):

    def __init__(self, factors, precompute):
        self.factors = factors
        self.variable_neighbors = {}
        self.precompute = precompute
        if precompute:
            # Compute variable neighbors in one pass over factors
            # to avoid looping over all factors for all variables.
            for f in factors:
                for v in f.variables:
                    if self.variable_neighbors.get(v) is None:
                        self.variable_neighbors[v] = [f]
                    else:
                        self.variable_neighbors[v] += [f]

    def get_var_neighbors(self, var: Variable):
        if self.precompute:
            return self.variable_neighbors.get(var, [])
        else:
            return [f for f in self.factors if var in f.variables]

    def neighbors(self, node):
        assert(isinstance(node, Factor) or isinstance(node, Variable))
        return (node.variables if (isinstance(node, Factor))
                else self.get_var_neighbors(node))


class SpanningTree(Graph):

    def __init__(self, graph: Graph):
        self.graph = graph
        self.children = {}
        self.parents = {}

    def neighbors(self, node):
        if not self.children.get(node):
            neighbors = [n for n in self.graph.neighbors(node) if n not in self.parents]
            for n in neighbors:
                self.parents[n] = node
            self.children[node] = neighbors
        return self.children[node]

    def evaluate(self, root, is_leaf=lambda x: False):
        def _evaluate(node):
            if not is_leaf(node) and node not in self.children:
                for n in self.neighbors(node):
                    _evaluate(n)

        self.parents[root] = None
        _evaluate(root)

    def least_common_ancestor(self, nodes):
        def getSelfWithParents(n):
            res = []
            while n is not None:
                res += [n]
                n = self.parents.get(n)
            return res

        if len(nodes) == 1:
            return nodes[0]
        if len(nodes) == 0:
            return None
        anc1 = getSelfWithParents(nodes[0])
        common = reduce(lambda x, y: x & y, [set(getSelfWithParents(n)) for n in nodes[1:]])
        for ancestor in anc1:
            if ancestor in common:
                return ancestor
        raise Exception("Tree does not have a single root")
