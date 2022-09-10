from abc import ABC
from typing import Any

from neuralpp.experiments.experimental_inference.graph_analysis import Tree, PartialFactorSpanningTree, PartialTree
from neuralpp.util.util import argmax, empty, get_or_compute_and_put


class TreeComputation:
    """
    Defines a computation evaluated on a tree, with a cache that can be recomputed as children change.
    """

    def compute(self, node) -> Any:
        """
        This computes the value of the TreeComputation on a given node,
        without using the cache.

        This method should not be called recursively, in order to take advantage of caching.
        Instead, use get or call syntax (e.g. `self[child_node]`' or `self(child)`) to retrieve
        the value from the cache or recompute it as needed.
        """
        raise NotImplementedError()

    def __init__(self, tree: Tree):
        self.tree = tree
        self.result_dict = {}

    def __getitem__(self, node):
        return get_or_compute_and_put(self.result_dict, node, self.compute, key_getter=id)

    def invalidate(self, node):
        """
        Invalidate cached value for `node` and its ancestors.
        """
        while node is not None:
            if id(node) in self.result_dict:
                del self.result_dict[id(node)]
            node = self.tree.parent(node)


class PartialTreeComputation(TreeComputation, ABC):
    """
    A TreeComputation that provides a way to add new edges to the underlying tree
    while performing the necessary updates to derived values.
    """

    def bookkeep_values_in_path_to(self, node):
        """
        Method for updating tree values when new edges are added.

        The base behavior of this method is to invalidate the target node,
        which leads to values depending on it (the values on the path from the root
        to the node) to be re-calculated when needed.
        Subclasses may override this with more efficient updating schemes.
        """
        self.invalidate(node)

    def add_edge(self, parent, child):
        self.tree.add_edge(parent, child)
        self.bookkeep_values_in_path_to(parent)


class MaximumLeafValueComputation(TreeComputation):
    """
    A TreeComputation where the value of a
    leaf is provided by given function leaf_value_function,
    and the value of each non-terminal node
    is the value among the values of its children
    that products the maximum value
    is argmax_{v in {value(child) for child in children(node)}} argmax_key(v)
    """

    def __init__(self, tree: Tree, leaf_value_function, argmax_key=None):
        super().__init__(tree)
        self.leaf_value_function = leaf_value_function
        self.argmax_key = argmax_key

    def compute(self, node):
        children = self.tree.children(node)
        if empty(children):
            return self.leaf_value_function(node, self.tree)
        else:
            candidates = [self[child] for child in children if self[child] is not None]
            return argmax(candidates, self.argmax_key)
