from neuralpp.inference.graphical_model.representation.factor.directed.graph.edge import (
    Edge,
)


class CycleFound(Exception):
    def __init__(self, cycle):
        self.cycle = cycle

    def __repr__(self):
        return f"Cycle found: {cycle}"


def make_cycle(path, counter_path):
    """
    Takes two paths, where a path is a list of distinct nodes, with inverted first and last nodes,
    and makes a list representing the cycle they form.
    """
    return path[:-1] + counter_path


def make_aggregated_edges_when_eliminating_variable(edges, variable):
    """
    Given a set of edges, each annotated with a set of paths originating it,
    returns a new set of edges equal to the original one but for the following properties:
    - the new set contains all original edges except for those inciding on variable.
    - for each pair (parent, variable) and (variable, child),
      ensures there is an edge (parent, child) in the new set
      whose paths contain the paths in the original set (if it was already present)
      plus the path (parent, variable, child).
    This ensures that, if we invoke this function on an initial set of edges,
    the current set of edges maintains the same connectivity of the original
    without containing previously eliminated variables, and the annotated paths
    indicate the origin of each edge in the original graph (original set of edges).
    Moreover, if a cycle is detected, a CycleFound exception is thrown with found cycle.
    """
    edges_to_children_of_variable = [e for e in edges if e.parent is variable]
    edges_from_parents_of_variable = [e for e in edges if e.child is variable]
    aggregated_edges_so_far = {
        (e.parent, e.child): e for e in edges if variable not in e
    }
    for edge_to_child in edges_to_children_of_variable:
        for edge_from_parent in edges_from_parents_of_variable:
            key = (edge_from_parent.parent, edge_to_child.child)
            if key in aggregated_edges_so_far:
                pass  # we already have this edge and a path on it, even if it is not the same path we just found.
            else:
                # edge is new, so we need to register it and its path if it does not close a cycle
                path = edge_from_parent.path[:-1] + edge_to_child.path
                check_there_is_no_cycle(key, path, aggregated_edges_so_far)
                aggregated_edges_so_far[key] = Edge(
                    edge_from_parent.parent, edge_to_child.child, path
                )
    aggregated_edges = aggregated_edges_so_far
    return aggregated_edges


def check_there_is_no_cycle(key, path, aggregated_edges_so_far):
    inverse_key = (key[1], key[0])
    inverse_edge = aggregated_edges_so_far.get(inverse_key)
    if inverse_edge is not None:
        raise CycleFound(make_cycle(path, inverse_edge.path))
