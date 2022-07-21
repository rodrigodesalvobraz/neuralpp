from neuralpp.experiments.experimental_inference.graph_analysis import LazyFactorSpanningTree, FactorGraph, FactorTree
from neuralpp.inference.graphical_model.representation.factor.product_factor import ProductFactor
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util import util


class BeliefPropagation:

    def __init__(self, tree: FactorTree):
        self.tree = tree

    def run(self):
        return self.message_from(self.tree.root).normalize()

    def product_at(self, node):
        incoming_messages = [self.message_from(n) for n in self.tree.children(node)]
        return ProductFactor.multiply(self.factor_at(node) + incoming_messages)

    @staticmethod
    def factor_at(node):
        return [] if isinstance(node, Variable) else [node]

    def variables_summed_out_at(self, node, all_variables_in_product_at_node):
        return util.subtract(
            all_variables_in_product_at_node,
            self.tree.external_variables(node)
        )

    def message_from(self, node):
        product_at_node = self.product_at(node)
        vars_summed_out = self.variables_summed_out_at(node, product_at_node.variables)
        return product_at_node ^ vars_summed_out


class ExactBeliefPropagation(BeliefPropagation):
    def __init__(self, factors, query):
        super().__init__(LazyFactorSpanningTree(FactorGraph(factors), query))
