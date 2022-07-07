from typing import Iterable, Set

from neuralpp.inference.graphical_model.representation.factor.product_factor import Factor
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util import util


class Graph:
    def neighbors(self, node):
        raise NotImplemented()


class FactorGraph(Graph):

    def __init__(self, factors):
        self.factors = factors
        self.variable_neighbors = {}
        # Compute variable neighbors in one pass over factors
        # to avoid looping over all factors for all variables.
        for f in factors:
            for v in f.variables:
                if self.variable_neighbors.get(v) is None:
                    self.variable_neighbors[v] = [f]
                else:
                    self.variable_neighbors[v] += [f]

    def neighbors(self, node):
        assert (isinstance(node, Factor) or isinstance(node, Variable))
        return (node.variables if (isinstance(node, Factor))
                else self.variable_neighbors.get(node, []))


class PartialSpanningTree:

    def __init__(self, graph: Graph, root, stop=lambda _: False):
        self.graph = graph
        self.root = root
        self._children = {}
        self._parents = {}

    def children(self, node):
        if not self._children.get(id(node)):
            neighbors = [n for n in self.graph.neighbors(node) if id(n) not in self._parents]
            for n in neighbors:
                self._parents[id(n)] = node
            self._children[id(node)] = neighbors
        return self._children[id(node)]

    def parent(self, node):
        return self._parents.get(id(node), None)


def node_variables(node):
    return node.variables if isinstance(node, Factor) else [node]


class FactorPartialSpanningTree(PartialSpanningTree):

    def __init__(self, graph: FactorGraph, root):
        super().__init__(graph, root)

        def _evaluate(node):
            for n in self.children(node):
                _evaluate(n)
        self._parents[id(root)] = None
        _evaluate(root)  # Currently evaluates a full spanning tree

    def variables(self, node) -> Iterable[Variable]:
        """ All variables appearing in the subtree rooted at node. """
        return util.union(
            [
                node_variables(node),
                util.union([self.variables(child)
                            for child in self.children(node)])
            ])

    def siblings_variables(self, node) -> Set[Variable]:
        """ Variables appearing in the subtree of at least one sibling of node """
        if self.parent(node) is None:
            return set([])
        else:
            return util.union([self.variables(sibling)
                               for sibling in self.children(self.parent(node))
                               if sibling is not node])

    def variables_in_node_and_ancestors(self, node) -> Set[Variable]:
        """ The set of variables defined by the current node and direct ancestors """
        def _ancestor_variables(n, result_set):
            if isinstance(n, Variable):
                result_set.add(n)
            parent = self.parent(n)
            if parent is None:
                return result_set
            return _ancestor_variables(parent, result_set)
        return _ancestor_variables(node, set([]))

    def external_variables(self, node) -> Set[Variable]:
        """ Variables appearing outside subtree of node (cousins' variables) """
        if self.parent(node) is None:
            return set([])
        else:
            return self.siblings_variables(node) | self.external_variables(self.parent(node))
