from typing import List, Optional

from neuralpp.experiments.experimental_inference.tree import FactorGraph, SpanningTree
from neuralpp.inference.graphical_model.representation.factor.product_factor import (
    Factor, ProductFactor
)
from neuralpp.inference.graphical_model.variable.variable import Variable
from neuralpp.util.group import Group


class ExactBeliefPropagation:

    def __init__(self, factors: List[Factor]):
        self.tree = SpanningTree(FactorGraph(factors))
        # identifies the least subtree which contains all usages of the key variable
        self.variable_roots = {}

    def run(self, query: Variable):
        self.tree.evaluate(query)
        return self.incoming_messages(query)

    def incoming_messages(self, node):
        messages = [self.incoming_messages(v) for v in self.tree.neighbors(node)]
        if isinstance(node, Variable):
            self.store_root_of_variable_usages(node)
            if not messages:
                return Group.identity
        else:
            messages.append(node)
        return ProductFactor.multiply(messages).atomic_factor() ^ self.variable_roots.get(node, [])

    def store_root_of_variable_usages(self, variable: Variable):
        children = set(self.tree.neighbors(variable))
        lca = self.tree.least_common_ancestor([n for n in self.tree.graph.neighbors(variable) if n not in children])
        if self.variable_roots.get(lca) is None:
            self.variable_roots[lca] = [variable]
        else:
            self.variable_roots[lca].append(variable)
