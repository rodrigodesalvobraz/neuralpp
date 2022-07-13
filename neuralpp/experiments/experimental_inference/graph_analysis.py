from typing import Iterable, Set

from neuralpp.inference.graphical_model.representation.factor.product_factor import Factor
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util import util
from neuralpp.util.cache_by_id import lru_cache_by_id


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

    def __init__(self, graph: Graph, root):
        self.graph = graph
        self.root = root
        self._children = {}
        self._parents = {}

    def children(self, node):
        if not self._children.get(id(node)):
            available_neighbors = [n for n in self.graph.neighbors(node) if id(n) not in self._parents]
            for n in available_neighbors:
                self._parents[id(n)] = node
            self._children[id(node)] = available_neighbors
        return self._children[id(node)]

    def parent(self, node):
        return self._parents.get(id(node), None)


def node_variables(node):
    """All variables which are used directly by a node"""
    return node.variables if isinstance(node, Factor) else [node]


def variable_at(node):
    return node if isinstance(node, Variable) else None


class FactorPartialSpanningTree(PartialSpanningTree):

    def __init__(self, graph: FactorGraph, root):
        super().__init__(graph, root)
        self._parents[id(root)] = None

    @lru_cache_by_id(1000)
    def variables(self, node) -> Iterable[Variable]:
        """ All variables appearing in the subtree rooted at node. """
        return util.union(
            [
                node_variables(node),
                util.union([self.variables(child)
                            for child in self.children(node)])
            ])

    @lru_cache_by_id(1000)
    def siblings_variables(self, node) -> Set[Variable]:
        """ Variables appearing in the subtree of at least one sibling of node """
        if self.parent(node) is None:
            return set([])
        else:
            return util.union([self.variables(sibling)
                               for sibling in self.children(self.parent(node))
                               if sibling is not node])

    @lru_cache_by_id(1000)
    def external_variables(self, node) -> Set[Variable]:
        """ Variables appearing outside subtree of node"""

        def local_external_variables(n):
            result = self.siblings_variables(n)
            node_variable = node if isinstance(node, Variable) else None
            if node_variable is not None:
                result.add(node_variable)
            return result

        if self.parent(node) is None:
            var_if_exists = variable_at(node)
            if var_if_exists is not None:
                return {var_if_exists}
            else:
                return set([])
        else:
            return local_external_variables(node) | self.external_variables(self.parent(node))
