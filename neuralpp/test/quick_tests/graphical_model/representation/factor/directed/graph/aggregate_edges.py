import pytest

from neuralpp.inference.graphical_model.representation.factor.directed.graph.aggregate_edges import \
    make_aggregated_edges_when_eliminating_variable, CycleFound
from neuralpp.inference.graphical_model.representation.factor.directed.graph.edge import Edge


def test_basic():
    # simple linear
    edges = {Edge(1, 2), Edge(2, 3)}
    variable = 2
    expected = {Edge(1, 3, (1, 2, 3))}
    run_test(edges, variable, expected)

    # sole edge
    edges = {Edge(1, 2)}
    variable = 2
    expected = set()
    run_test(edges, variable, expected)

    # none
    edges = set()
    variable = 2
    expected = set()
    run_test(edges, variable, expected)

    # diamond
    edges = {Edge(1, 2), Edge(1, 3), Edge(2, 4), Edge(3, 4)}
    variable = 2
    expected = {Edge(1, 4, (1, 2, 4)), Edge(1, 3), Edge(3, 4)}
    run_test(edges, variable, expected)

    # funnel
    edges = {Edge(1, 3), Edge(2, 3), Edge(3, 4), Edge(3, 5)}
    variable = 3
    expected = {Edge(1, 4, (1, 3, 4)), Edge(1, 5, (1, 3, 5)), Edge(2, 4, (2, 3, 4)), Edge(2, 5, (2, 3, 5))}
    run_test(edges, variable, expected)

    # funnel with extra edges
    edges = {Edge(0, 1), Edge(1, 3), Edge(2, 3), Edge(3, 4), Edge(3, 5), Edge(5, 6),
             Edge(123, 432)}
    variable = 3
    expected = {Edge(0, 1), Edge(1, 4, (1, 3, 4)), Edge(1, 5, (1, 3, 5)), Edge(2, 4, (2, 3, 4)), Edge(2, 5, (2, 3, 5)),
                Edge(5, 6), Edge(123, 432)}
    run_test(edges, variable, expected)


def test_cycles():
    edges = {Edge(0, 1), Edge(1, 2), Edge(2, 0), Edge(2, 3), Edge(1, 4)}
    variable = 1
    with pytest.raises(CycleFound, match="(0, 1, 2, 0)"):
        run_test(edges, variable, expected=None)


def run_test(edges, variable, expected):
    actual = set(make_aggregated_edges_when_eliminating_variable(edges, variable).values())
    assert expected == actual
