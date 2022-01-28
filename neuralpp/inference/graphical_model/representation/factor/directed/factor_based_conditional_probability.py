class FactorBasedConditionalProbability(AbstractConditionalProbability):

    def __init__(self, factor, children):
        self.factor = factor
        self.children = children
        self.parents = [v for v in factor.variables if v not in children]

    def aggregate(self, variable):
        all_edges = [e for e in factor.edges
                     for factor in ProductFactors.factors(self.factor)
                     if isinstance(factor, ConditionalProbability)]
        children_of_variable = [c for (p, c) in all_edges if p is variable]
        parents_of_variable = [p for (p, c) in all_edges if c is variable]
        edges_without_variable = [e for e in all_edges if variable not in e]