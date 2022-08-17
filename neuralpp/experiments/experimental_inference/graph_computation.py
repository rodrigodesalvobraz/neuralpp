from neuralpp.util.util import argmax


class PartialTreeComputation:
    """
    Defines a computation evaluated on partial_tree, which can be updated
    as the partial tree gains edges from the full tree.

    Requires that partial_tree is a subtree of full_tree. (TODO: Add validation for this)

    * Get syntax is used to retrieve cached values of the computation.
    * Call syntax will evaluate the function over the partial tree, updating the result_dict. (TODO: do we need this?
      in most cases it's only used during initialization, as long as results are propagated when changed.)
    """

    def __init__(self, partial_tree, full_tree):
        self.partial_tree = partial_tree
        self.full_tree = full_tree
        self.result_dict = {}

    def __getitem__(self, item):
        return self.result_dict[id(item)]

    def __setitem__(self, key, value):
        self.result_dict[id(key)] = value

    def __call__(self, item):
        self.compute_result_dict(item)
        return self.result_dict[id(item)]

    def compute_result_dict(self, node):
        raise NotImplementedError()

    def update_value(self, target_node, changed_child, child_value):
        raise NotImplementedError()

    def recompute_and_propagate_result_to_parents(self, node):
        self.compute_result_dict(node)
        node_result = self[node]
        target_node = self.partial_tree.parent(node)
        while target_node is not None:
            self.update_value(target_node, node, node_result)
            node = target_node
            target_node = self.partial_tree.parent(target_node)


class ExpansionValueComputation(PartialTreeComputation):

    def __init__(self, partial_tree, full_tree, expansion_value_fn):
        super().__init__(partial_tree, full_tree)
        self.expansion_value_fn = expansion_value_fn
        self.compute_result_dict(partial_tree.root)

    def compute_result_dict(self, node):
        if node not in self.partial_tree:
            result = (node, self.expansion_value_fn(node, self.partial_tree, self.full_tree))
        else:
            children = self.full_tree.children(node)
            if children:
                for child in children:
                    self.compute_result_dict(child)
                result = argmax(
                    (self[child] for child in children),
                    lambda node_value_pair: node_value_pair[1],
                )
            else:
                result = None
        self[node] = result

    def update_value(self, target_node, changed_child, child_value):
        ancestor_value = self.result_dict.get(id(target_node))
        if child_value is None:
            # Current node is finished expanding.
            # Attempt to replace ancestors' expansion value with the highest available child
            update_candidates = [
                self[child] for child in self.full_tree.children(target_node)
                if self[child] is not None
            ]
            if update_candidates:
                child_value = argmax(
                    update_candidates,
                    lambda node_value_pair: node_value_pair[1],
                )
            self[target_node] = child_value
        elif ancestor_value is not None and child_value[1] <= ancestor_value[1]:
            # If the updated value at node is already higher than any other value in the ancestor's subtree,
            # we don't need to look at any other children. Otherwise, we need to check these values:
            update_candidates = [
                self[child] for child in self.full_tree.children(target_node)
                if self[child] is not None
            ]
            update_candidates.append(child_value)
            child_value = argmax(
                update_candidates,
                lambda node_value_pair: node_value_pair[1],
            )
        self[target_node] = child_value

    def expand_partial_tree_and_recompute(self, expansion_root):
        expand_to_node = self[expansion_root][0]
        parent = self.full_tree.parent(expand_to_node)
        assert(parent is not None)
        self.partial_tree.add_edge(parent, expand_to_node)

        # Compute available children's expansion results and update value of expanded node
        # to the highest available child result, then propagate through ancestors:
        self.recompute_and_propagate_result_to_parents(expand_to_node)

