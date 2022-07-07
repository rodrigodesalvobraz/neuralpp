from neuralpp.experiments.experimental_inference.graph_analysis import FactorPartialSpanningTree, FactorGraph
from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    Factor, ProductFactor
)
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util import util
from neuralpp.util.group import Group


class BeliefPropagation:

    def __init__(self, tree: FactorPartialSpanningTree):
        self.tree = tree

    def run(self):
        return self.message_from(self.tree.root)

    def product_at(self, node):
        incoming_messages = [self.message_from(n) for n in self.tree.children(node)]
        return ProductFactor.multiply(self.factor_at(node) + incoming_messages)

    @staticmethod
    def factor_at(node):
        return [] if isinstance(node, Variable) else [node]

    def variables_summed_out_at(self, node, incoming_variables):
        return util.subtract(
            incoming_variables,
            self.tree.external_variables(node) | self.tree.variables_in_node_and_ancestors(node))

    def message_from(self, node):
        product_at_node = self.product_at(node).atomic_factor()
        if product_at_node is Group.identity:
            return product_at_node
        vars_summed_out = self.variables_summed_out_at(node, product_at_node.variables)
        return product_at_node ^ vars_summed_out


class ExactBeliefPropagation(BeliefPropagation):
    def __init__(self, factors, query):
        super().__init__(FactorPartialSpanningTree(FactorGraph(factors), query))
