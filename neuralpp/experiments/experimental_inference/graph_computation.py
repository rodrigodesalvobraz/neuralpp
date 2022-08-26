from neuralpp.experiments.experimental_inference.graph_analysis import Tree, PartialFactorSpanningTree
from neuralpp.util.util import argmax, empty


class TreeComputation:
    """
    Defines a computation evaluated on a tree, with a cache that can be recomputed as children change.
    """

    def compute(self, node):
        """
        This defines the value of the TreeComputation on a given node.

        `compute` should not be called recursively, in order to take advantage of caching.
        Instead, use get or call syntax (e.g. `self[child_node]`' or `self(child)`) to retrieve
        the value from the cache or recompute it as needed.
        """
        raise NotImplementedError()

    def update_value(self, target_node):
        """
        The base behavior of this method is to invalidate the target node and recompute
        over it and its ancestors.
        """
        self.invalidate(target_node)
        self.compute_result_dict(self.tree.root)

    def __init__(self, tree: Tree):
        self.tree = tree
        self.result_dict = {}

    def __contains__(self, item):
        return id(item) in self.result_dict

    def __getitem__(self, item):
        if item in self:
            return self.result_dict[id(item)]
        self.compute_result_dict(item)
        return self.result_dict[id(item)]

    def __setitem__(self, item, value):
        self.result_dict[id(item)] = value

    def __call__(self, item):
        return self[item]

    def compute_result_dict(self, node):
        self[node] = self.compute(node)

    def invalidate(self, node):
        """
        Invalidate cached value for `node` and its ancestors.
        """
        while node is not None:
            if node in self:
                del self.result_dict[id(node)]
            node = self.tree.parent(node)


class MaximumLeafValueComputation(TreeComputation):

    def __init__(self, tree: Tree, leaf_value_function, argmax_key=None):
        super().__init__(tree)
        self.leaf_value_function = leaf_value_function
        self.argmax_key = argmax_key
        self.compute_result_dict(tree.root)

    def compute(self, node):
        children = self.tree.children(node)
        if empty(children):
            return self.leaf_value_function(node, self.tree)
        else:
            candidates = [self[child] for child in children if self[child] is not None]
            return argmax(candidates, self.argmax_key)
