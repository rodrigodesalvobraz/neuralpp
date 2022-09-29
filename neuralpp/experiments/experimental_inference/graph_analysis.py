from collections import defaultdict
from typing import Iterable, Set

from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    Factor,
)
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util import util


# TODO: Incremental updates for variable processing, probably using a PartialTreeComputation.
#       Expanding the partial tree results in introducing more variables.


class Graph:
    def neighbors(self, node):
        raise NotImplemented()

    def __contains__(self, item):
        raise NotImplemented()

    def contains_edge(self, parent, child):
        raise NotImplemented()


class FactorGraph(Graph):
    def __init__(self, factors):
        self.factors = factors
        self.variable_neighbors = defaultdict(list)
        # Compute variable neighbors in one pass over factors
        # to avoid looping over all factors for all variables.
        for f in factors:
            for v in f.variables:
                self.variable_neighbors[v].append(f)

    def neighbors(self, node):
        assert isinstance(node, Factor) or isinstance(node, Variable)
        return (
            node.variables
            if (isinstance(node, Factor))
            else self.variable_neighbors[node]
        )

    @staticmethod
    def factor_at(node):
        return [] if isinstance(node, Variable) else [node]

    @staticmethod
    def variables_at(node):
        """All variables which are used directly by a node."""
        return node.variables if isinstance(node, Factor) else [node]


class Tree(Graph):
    def __init__(self):
        self.root = None
        self.depth_cache = {}

    def children(self, node):
        raise NotImplemented()

    def parent(self, node):
        raise NotImplemented()

    def contains_edge(self, parent, child):
        return self.parent(child) == parent

    def _compute_depth(self, node):
        return 0 if self.parent(node) is None else self.depth(self.parent(node)) + 1

    def depth(self, node):
        return util.get_or_compute_and_put(
            self.depth_cache, node, self._compute_depth, key_getter=id
        )


class PartialTree(Tree):
    def __init__(self, *args):
        super().__init__()

    @property
    def full_tree(self) -> Tree:
        raise NotImplemented


class LazySpanningTree(Tree):
    def __init__(self, graph: Graph, root):
        super().__init__()
        self.graph = graph
        self.root = root
        self._children = {}
        self._parents = {id(root): None}

    def __contains__(self, item):
        return id(item) in self._parents

    def children(self, node):
        if self._children.get(id(node)) is None:
            available_neighbors = [
                n for n in self.graph.neighbors(node) if id(n) not in self._parents
            ]
            for n in available_neighbors:
                self._parents[id(n)] = node
            self._children[id(node)] = available_neighbors
        return self._children[id(node)]

    def parent(self, node):
        return self._parents.get(id(node), None)


class FactorTree(Tree, FactorGraph):
    """
    A tree of factors that also provides the external variables for each subtree (identified by its root).
    The external variables of a subtree are the variables that appear somewhere in the whole tree
    outside the subtree, plus the variables appearing at the subtree's root node.
    """

    def external_variables(self, node) -> Set[Variable]:
        raise NotImplemented()


class LazyFactorSpanningTree(LazySpanningTree, FactorTree):
    def variables(self, node) -> Iterable[Variable]:
        """All variables appearing in the subtree rooted at node."""
        return util.union(
            [
                self.variables_at(node),
                util.union([self.variables(child) for child in self.children(node)]),
            ]
        )

    def siblings_variables(self, node) -> Set[Variable]:
        """Variables appearing in the subtree of at least one sibling of node"""
        if self.parent(node) is None:
            return set()
        else:
            return util.union(
                self.variables(sibling)
                for sibling in self.children(self.parent(node))
                if sibling is not node
            )

    def external_variables(self, node) -> Set[Variable]:
        """Variables appearing outside subtree of node"""

        def local_external_variables(n):
            return self.siblings_variables(n) | set(self.variables_at(node))

        if self.parent(node) is None:
            return set(self.variables_at(node))
        else:
            return local_external_variables(node) | self.external_variables(
                self.parent(node)
            )


class PartialFactorSpanningTree(LazyFactorSpanningTree, PartialTree):
    # Unlike the base LazyFactorSpanningTree, the partial spanning tree only returns possible children
    # when the edge has been explicitly added.

    def __init__(self, full_tree: FactorTree):
        super().__init__(full_tree, full_tree.root)

    @property
    def full_tree(self):
        return self.graph

    def children(self, node):
        return self._children.get(id(node), [])

    def add_edge(self, parent, child):
        assert self.graph.contains_edge(parent, child)
        if child in self:
            raise Exception(f"Child node {child} has already been added to the tree")
        elif id(parent) in self._children:
            self._children[id(parent)].append(child)
        else:
            self._children[id(parent)] = [child]
        self._parents[id(child)] = parent


class PartialTreePlusOneLevel(Tree):
    """
    This represents the subtree formed by a partial tree + any nodes in the full tree which
    have parents in the partial tree. If the base subtree is changed, the partial expansion tree
    will likewise be affected.
    """

    def __init__(self, partial_tree: PartialTree):
        super().__init__()
        self.partial_tree = partial_tree
        self.root = self.partial_tree.root

    def __contains__(self, item):
        return item in self.partial_tree or self.parent(item) is not None

    def children(self, node):
        if node in self.partial_tree:
            return self.partial_tree.full_tree.children(node)
        else:
            return []

    def parent(self, node):
        full_tree_parent = self.partial_tree.full_tree.parent(node)
        if full_tree_parent not in self.partial_tree:
            return None
        return full_tree_parent

    def contains_edge(self, parent, child):
        return parent in self.partial_tree or child == self.partial_tree.root
