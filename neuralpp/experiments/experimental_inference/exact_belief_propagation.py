from typing import List, Iterable

from neuralpp.experiments.experimental_inference.tree import FactorGraph, SpanningTree
from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    Factor, ProductFactor
)
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util.group import Group


class ExactBeliefPropagation:

    def __init__(self, factors: List[Factor], precompute):
        self.tree = SpanningTree(FactorGraph(factors, precompute))
        self.sum_points = {}

    def run(self, query: Variable):
        self.tree.evaluate(query)
        return self.msg_from_node(query)

    def msg_from_node(self, node):
        messages = [self.msg_from_node(v) for v in self.tree.neighbors(node)]
        if isinstance(node, Variable):
            self.process_variable_sum_point(node)
            if not messages:
                return Group.identity
        else:
            messages += [node]
        return ProductFactor.multiply(messages).atomic_factor().sum_out_variables(self.sum_points.get(node, []))

    def process_variable_sum_point(self, variable: Variable):
        children = set(self.tree.neighbors(variable))
        lca = self.tree.least_common_ancestor([n for n in self.tree.graph.neighbors(variable) if n not in children])
        if self.sum_points.get(lca) is None:
            self.sum_points[lca] = [variable]
        else:
            self.sum_points[lca] += [variable]
