from collections import namedtuple

from neuralpp.experiments.experimental_inference.graph_analysis import (
    LazyFactorSpanningTree,
    FactorGraph,
    FactorTree,
    PartialFactorSpanningTree,
    PartialTreePlusOneLevel,
)
from neuralpp.experiments.experimental_inference.graph_computation import (
    MaximumLeafValueComputation,
    PartialTreeComputation,
)
from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    ProductFactor,
)
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
            all_variables_in_product_at_node, self.tree.external_variables(node)
        )


class ExactBeliefPropagation(BeliefPropagation):
    def __init__(self, factors, query):
        super().__init__(LazyFactorSpanningTree(FactorGraph(factors), query))


Expansion = namedtuple("Expansion", "node expansion_value")


class AnytimeExactBeliefPropagation(PartialTreeComputation):
    def __init__(
        self, partial_tree, full_tree, approximation, expansion_value_function
    ):
        super().__init__(partial_tree)
        self.approximation = approximation
        self.full_tree = full_tree
        self.expansion = MaximumLeafValueComputation(
            PartialTreePlusOneLevel(partial_tree),
            lambda node, tree: Expansion(
                node, expansion_value_function(node, self.tree, self.full_tree)
            )
            if node not in partial_tree
            else None,
            lambda node_value_pair: node_value_pair.expansion_value,
        )

    @staticmethod
    def from_factors(factors, query, approximation, expansion_value_function):
        """
        Simple utility to start AnytimeExactBeliefPropagation on a partial tree containing just the root.
        """
        full_tree = LazyFactorSpanningTree(FactorGraph(factors), query)
        partial_tree = PartialFactorSpanningTree(full_tree)
        return AnytimeExactBeliefPropagation(
            partial_tree, full_tree, approximation, expansion_value_function
        )

    def run(self):
        return self[self.tree.root].normalize()

    def compute(self, node):
        if node not in self.tree:
            return self.approximation(node, self.tree, self.full_tree)
        product_at_node = self.product_at(node)
        vars_summed_out = self.variables_summed_out_at(node, product_at_node.variables)
        result = product_at_node ^ vars_summed_out
        return result

    def product_at(self, node):
        incoming_messages = [self.compute(n) for n in self.full_tree.children(node)]
        return ProductFactor.multiply(self.tree.factor_at(node) + incoming_messages)

    def variables_summed_out_at(self, node, all_variables_in_product_at_node):
        return util.subtract(
            all_variables_in_product_at_node, self.tree.external_variables(node)
        )

    def expand(self, expansion_root):
        potential_expansion = self.expansion[expansion_root]
        if potential_expansion is None:
            return
        expand_to_node = potential_expansion.node
        parent = self.full_tree.parent(expand_to_node)
        self.add_edge(parent, expand_to_node)
        self.expansion.invalidate(expand_to_node)

    def is_complete(self):
        return self.expansion[self.tree.root] is None
