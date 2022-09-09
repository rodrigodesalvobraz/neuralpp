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

    def update_value(self, target_node):
        """
        Method for updating tree values when new edges are added.

        The base behavior of this method is to invalidate the target node. The updated value will
        be computed lazily from `compute`. Subclasses may override this with more efficient updating
        schemes.
        """
        self.invalidate(target_node)

    def add_edge(self, parent, child):
        self.tree.add_edge(parent, child)
        self.update_value(parent)


class MaximumLeafValueComputation(TreeComputation):

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
