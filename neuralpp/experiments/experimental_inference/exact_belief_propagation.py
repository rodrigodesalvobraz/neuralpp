from neuralpp.experiments.experimental_inference.graph_analysis import LazyFactorSpanningTree, FactorGraph, FactorTree, \
    PartialFactorSpanningTree
from neuralpp.experiments.experimental_inference.graph_computation import ExpansionValueComputation, \
    TreeComputation
from neuralpp.inference.graphical_model.representation.factor.product_factor import ProductFactor
from neuralpp.util import util


class BeliefPropagation:

    def __init__(self, tree: FactorTree):
        self.tree = tree

    def run(self):
        return self.message_from(self.tree.root).normalize()

    def message_from(self, node):
        product_at_node = self.product_at(node)
        vars_summed_out = self.variables_summed_out_at(node, product_at_node.variables)
        return product_at_node ^ vars_summed_out

    def product_at(self, node):
        incoming_messages = [self.message_from(n) for n in self.tree.children(node)]
        return ProductFactor.multiply(self.tree.factor_at(node) + incoming_messages)

    def variables_summed_out_at(self, node, all_variables_in_product_at_node):
        return util.subtract(
            all_variables_in_product_at_node,
            self.tree.external_variables(node)
        )


class ExactBeliefPropagation(BeliefPropagation):
    def __init__(self, factors, query):
        super().__init__(LazyFactorSpanningTree(FactorGraph(factors), query))


class AnytimeExactBeliefPropagation(TreeComputation):

    def __init__(self, partial_tree, full_tree, approximation_fn, expansion_fn):
        super().__init__(partial_tree)
        self.approximation = approximation_fn
        self.expansion = ExpansionValueComputation(partial_tree, full_tree, expansion_fn)
        self.compute_result_dict(partial_tree.root)

    @property
    def full_tree(self):
        return self.expansion.full_tree

    @staticmethod
    def from_factors(factors, query, approximation, expansion_value_function):
        """
        Simple utility to start AnytimeExactBeliefPropagation on a partial tree containing just the root.
        """
        full_tree = LazyFactorSpanningTree(FactorGraph(factors), query)
        partial_tree = PartialFactorSpanningTree(full_tree)
        return AnytimeExactBeliefPropagation(partial_tree, full_tree, approximation, expansion_value_function)

    def compute_result_dict(self, node):
        self[node] = None
        self.message_from(node)

    def update_value(self, target_node, child_value):
        self.compute_result_dict(target_node)

    def run(self):
        return self[self.tree.root].normalize()

    def message_from(self, node):
        if node not in self.tree:
            return self.approximation(node, self.tree, self.full_tree)
        cached = self.result_dict.get(id(node))
        if cached is not None:
            return cached
        product_at_node = self.product_at(node)
        vars_summed_out = self.variables_summed_out_at(node, product_at_node.variables)
        result = product_at_node ^ vars_summed_out
        self[node] = result
        return self[node]

    def product_at(self, node):
        incoming_messages = [self.message_from(n) for n in self.full_tree.children(node)]
        return ProductFactor.multiply(self.tree.factor_at(node) + incoming_messages)

    def variables_summed_out_at(self, node, all_variables_in_product_at_node):
        return util.subtract(
            all_variables_in_product_at_node,
            self.tree.external_variables(node)
        )

    def expand_partial_tree_and_recompute(self, expansion_root):
        potential_expansion = self.expansion[expansion_root]
        if potential_expansion is None:
            return
        self.expansion.expand_partial_tree_and_recompute(expansion_root)
        self.recompute_and_propagate_result_to_ancestors(potential_expansion.node)

    def is_complete(self):
        return self.expansion.is_complete()

