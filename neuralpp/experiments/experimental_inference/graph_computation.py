from collections import namedtuple

from neuralpp.experiments.experimental_inference.graph_analysis import Tree, PartialFactorSpanningTree
from neuralpp.util.util import argmax


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


Expansion = namedtuple("Expansion", "node expansion_value")


class ExpansionValueComputation(TreeComputation):

    def __init__(self, partial_tree: PartialFactorSpanningTree, full_tree, expansion_value_function):
        super().__init__(partial_tree)
        self.full_tree = full_tree
        self.expansion_value_function = expansion_value_function
        self.compute_result_dict(partial_tree.root)

    def compute(self, node):
        if node not in self.tree:
            return Expansion(node, self.expansion_value_function(node, self.tree, self.full_tree))
        else:
            children = self.full_tree.children(node)
            candidates = [self[child] for child in children if self[child] is not None]
            return argmax(candidates, lambda node_value_pair: node_value_pair.expansion_value)

    def expand_partial_tree_and_recompute(self, expansion_root):
        expand_to_node = self[expansion_root].node
        parent = self.full_tree.parent(expand_to_node)
        assert(parent is not None)
        self.tree.add_edge(parent, expand_to_node)

        # Compute available children's expansion results and update value of expanded node
        # to the highest available child result, then propagate through ancestors:
        self.update_value(expand_to_node)

    def is_complete(self):
        return all(self.result_dict[k] is None for k in self.result_dict)
