from collections import namedtuple

from neuralpp.util.util import argmax


class TreeComputation:
    """
    Defines a computation evaluated on a tree, with a cache that can be recomputed as children change.

    * Get syntax is used to retrieve cached values of the computation.
    * Call syntax will evaluate the function over the partial tree, updating the result_dict. (TODO: do we need this?
      in most cases it's only used during initialization, as long as results are propagated when changed.)
    """

    def __init__(self, tree):
        self.tree = tree
        self.result_dict = {}

    def __getitem__(self, item):
        return self.result_dict[id(item)]

    def __setitem__(self, item, value):
        self.result_dict[id(item)] = value

    def __call__(self, item):
        self.compute_result_dict(item)
        return self.result_dict[id(item)]

    def compute_result_dict(self, node):
        raise NotImplementedError()

    def update_value(self, target_node, child_value):
        raise NotImplementedError()

    def recompute_and_propagate_result_to_ancestors(self, node):
        self.compute_result_dict(node)
        node_result = self[node]
        target_node = self.tree.parent(node)
        while target_node is not None:
            self.update_value(target_node, node_result)
            target_node = self.tree.parent(target_node)


Expansion = namedtuple("Expansion", "node expansion_value")


class ExpansionValueComputation(TreeComputation):

    def __init__(self, partial_tree, full_tree, expansion_value_function):
        super().__init__(partial_tree)
        self.full_tree = full_tree
        self.expansion_value_function = expansion_value_function
        self.compute_result_dict(partial_tree.root)

    def __compute_single_node(self, node, get_child_scores_from_cache: bool):
        children = self.full_tree.children(node)
        if not get_child_scores_from_cache:
            for child in children:
                self.compute_result_dict(child)
        candidates = [self[child] for child in children if self[child] is not None]
        return argmax(candidates, lambda node_value_pair: node_value_pair.expansion_value)

    def compute_result_dict(self, node):
        if node not in self.tree:
            self[node] = Expansion(node, self.expansion_value_function(node, self.tree, self.full_tree))
        else:
            self[node] = self.__compute_single_node(node, False)

    def update_value(self, target_node, child_value):
        ancestor_value = self.result_dict.get(id(target_node))
        assert(ancestor_value is not None)
        if child_value is None or child_value.expansion_value <= ancestor_value.expansion_value:
            self[target_node] = self.__compute_single_node(target_node, True)
        else:
            self[target_node] = child_value

    def expand_partial_tree_and_recompute(self, expansion_root):
        expand_to_node = self[expansion_root].node
        parent = self.full_tree.parent(expand_to_node)
        assert(parent is not None)
        self.tree.add_edge(parent, expand_to_node)

        # Compute available children's expansion results and update value of expanded node
        # to the highest available child result, then propagate through ancestors:
        self.recompute_and_propagate_result_to_ancestors(expand_to_node)

    def is_complete(self):
        return all(self.result_dict[k] is None for k in self.result_dict)
