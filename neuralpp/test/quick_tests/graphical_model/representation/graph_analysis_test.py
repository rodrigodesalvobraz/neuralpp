from neuralpp.experiments.experimental_inference.exact_belief_propagation import Expansion
from neuralpp.experiments.experimental_inference.graph_analysis import *
from neuralpp.experiments.experimental_inference.graph_computation import *
from neuralpp.inference.graphical_model.representation.factor.factor import Factor
from neuralpp.inference.graphical_model.representation.factor.pytorch_table_factor import (
    PyTorchTableFactor,
)
from neuralpp.inference.graphical_model.variable.integer_variable import IntegerVariable


def test_ebp_tree_expansion():
    prob_cloudy = [0.2, 0.4, 0.4]
    prob_sprinkler = [0.6, 0.4]
    prob_rain_given_cloudy = [
        [0.6, 0.3, 0.1],
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
    ]

    def prob_wet_grass(wetness: int, rain_level: int, sprinkler_on: int):
        return 1.0 if (rain_level + sprinkler_on == wetness) else 0.0

    c = IntegerVariable("c", 3)
    r = IntegerVariable("r", 3)
    s = IntegerVariable("s", 2)
    w = IntegerVariable("w", 4)

    factors = [
        PyTorchTableFactor([c], prob_cloudy),
        PyTorchTableFactor([c, r], prob_rain_given_cloudy),
        PyTorchTableFactor([s], prob_sprinkler),
        PyTorchTableFactor.from_function([w, r, s], prob_wet_grass),
    ]

    # Prefers nodes with more variable names later in the alphabet
    def scoring_function(x, tree):
        if isinstance(x, IntegerVariable):
            return ord(x.name)
        else:
            assert(isinstance(x, Factor))
            return sum([ord(var.name) for var in x.variables])

    # A wrapper for testing convenience
    def scoring_result(x):
        return Expansion(x, scoring_function(x, None))

    full_tree = LazyFactorSpanningTree(FactorGraph(factors), w)
    partial_tree = PartialFactorSpanningTree(full_tree)

    expansion_computation = MaximumLeafValueComputation(
        PartialExpansionTree(partial_tree),
        lambda node, tree: Expansion(node, scoring_function(node, None)) if node not in partial_tree else None,
        lambda node_value_pair: node_value_pair.expansion_value
    )

    def expand_and_recompute():
        expanded_node = expansion_computation[partial_tree.root].node
        partial_tree.add_edge(full_tree.parent(expanded_node), expanded_node)
        expansion_computation.update_value(expanded_node)

    factor_wrs_value = scoring_result(factors[3])
    factor_s_value = scoring_result(factors[2])
    factor_cr_value = scoring_result(factors[1])
    factor_c_value = scoring_result(factors[0])
    variable_s_value = scoring_result(s)
    variable_r_value = scoring_result(r)
    variable_c_value = scoring_result(c)

    assert(len(expansion_computation.result_dict) == 2)
    assert(expansion_computation[w] == factor_wrs_value)
    assert(expansion_computation[factors[3]] == factor_wrs_value)

    assert(factors[3] not in partial_tree)
    expand_and_recompute()
    assert(factors[3] in partial_tree)

    # Two new nodes are checked in this expansion: r and s, both potential children of the factor on w, r, s
    # s has the higher priority by our alphabetical scoring function
    assert(len(expansion_computation.result_dict) == 4)
    assert(expansion_computation[w] == variable_s_value)
    assert(expansion_computation[factors[3]] == variable_s_value)
    assert(expansion_computation[s] == variable_s_value)
    assert(expansion_computation[r] == variable_r_value)

    assert(s not in partial_tree)
    expand_and_recompute()
    assert(s in partial_tree)

    # Factor on s is the highest value for each node except r (which is not its ancestor)
    assert(len(expansion_computation.result_dict) == 5)
    assert(expansion_computation[w] == factor_s_value)
    assert(expansion_computation[factors[3]] == factor_s_value)
    assert(expansion_computation[factors[2]] == factor_s_value)
    assert(expansion_computation[s] == factor_s_value)
    assert(expansion_computation[r] == variable_r_value)

    assert(factors[2] not in partial_tree)
    expand_and_recompute()
    assert(factors[2] in partial_tree)

    # Expansion on r now has the highest value after the path through s has been exhausted
    assert(len(expansion_computation.result_dict) == 5)
    assert(expansion_computation[w] == variable_r_value)
    assert(expansion_computation[r] == variable_r_value)
    assert(expansion_computation[factors[3]] == variable_r_value)
    assert(expansion_computation[factors[2]] is None)
    assert(expansion_computation[s] is None)

    assert(r not in partial_tree)
    expand_and_recompute()
    assert(r in partial_tree)

    assert(len(expansion_computation.result_dict) == 6)
    assert(expansion_computation[w] == factor_cr_value)
    assert(expansion_computation[r] == factor_cr_value)
    assert(expansion_computation[factors[3]] == factor_cr_value)
    assert(expansion_computation[factors[1]] == factor_cr_value)
    assert(expansion_computation[factors[2]] is None)
    assert(expansion_computation[s] is None)

    assert(factors[1] not in partial_tree)
    expand_and_recompute()
    assert(factors[1] in partial_tree)

    assert(len(expansion_computation.result_dict) == 7)
    assert(expansion_computation[w] == variable_c_value)
    assert(expansion_computation[r] == variable_c_value)
    assert(expansion_computation[c] == variable_c_value)
    assert(expansion_computation[factors[3]] == variable_c_value)
    assert(expansion_computation[factors[1]] == variable_c_value)
    assert(expansion_computation[factors[2]] is None)
    assert(expansion_computation[s] is None)

    assert(c not in partial_tree)
    expand_and_recompute()
    assert(c in partial_tree)

    assert(len(expansion_computation.result_dict) == 8)
    assert(expansion_computation[w] == factor_c_value)
    assert(expansion_computation[r] == factor_c_value)
    assert(expansion_computation[c] == factor_c_value)
    assert(expansion_computation[factors[3]] == factor_c_value)
    assert(expansion_computation[factors[1]] == factor_c_value)
    assert(expansion_computation[factors[0]] == factor_c_value)
    assert(expansion_computation[factors[2]] is None)
    assert(expansion_computation[s] is None)

    assert(factors[0] not in partial_tree)
    expand_and_recompute()
    assert(factors[0] in partial_tree)

    # All nodes have been fully expanded
    assert(len(expansion_computation.result_dict) == 8)
    for node_id in expansion_computation.result_dict:
        assert(expansion_computation.result_dict[node_id] is None)

