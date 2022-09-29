import random


def discrete_sample(points, get_probability):
    """
    Given a discrete set {e1, ... } and a probability function p,
    returns the first element in the set such that u < sum_i p(e_i)
    where u is uniformly sampled from [0; 1].
    This has the effect of sampling from the discrete set according to the probability function.
    """
    u = random.uniform(0, 1)
    total_probability_so_far = 0
    for point in points:
        total_probability_so_far += get_probability(point)
        if u < total_probability_so_far:
            return point
    raise Exception("Discrete probability provided for sampling is not normalized")
