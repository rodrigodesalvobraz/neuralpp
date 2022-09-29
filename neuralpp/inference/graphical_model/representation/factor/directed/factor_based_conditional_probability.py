from neuralpp.inference.graphical_model.representation.factor.directed.graph.aggregate_edges import (
    make_aggregated_edges_when_eliminating_variable,
)


class FactorBasedConditionalProbability(AbstractConditionalProbability):
    def __init__(self, factor, parents, children, edges):
        super(AbstractConditionalProbability, self).__init__(parents, children, edges)
        self.factor = factor

    def aggregate_edges_when_eliminating_variable(self, variable):
        """
        Returns the children, parents and edges of the conditional probability
        resulting from eliminating a variable from self.
        """
        aggregated_parents = util.subtract(self.parents, [variable])
        aggregated_children = util.subtract(self.children, [variable])
        aggregated_edges = make_aggregated_edges_when_eliminating_variable(
            self.edges, variable
        )
        return aggregated_parents, aggregated_children, aggregated_edges

    def __xor__(self, variable_or_variables):
        resulting_factor = self.factor ^ variable_or_variables
        aggregated_data = self.aggregate_edges_when_eliminating_variable(variable)
        return FactorBasedConditionalProbability(resulting_factor, *aggregated_data)
