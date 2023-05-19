import math
from dataclasses import dataclass
from numbers import Number
from typing import Iterable, Callable, Dict, TypeVar

T = TypeVar("T")


def add_to_histogram(histogram: Dict[T, int], value: T) -> None:
    """
    Increments count of value in histogram.
    histogram is a dict mapping values to their count.
    """
    histogram[value] = histogram.get(value, 0) + 1


def compute_histogram_dict(iterable: Iterable[T]) -> Dict[T, int]:
    """
    Creates a dict mapping elements in an iterable to their count.
    """
    histogram_dict = {}
    for value in iterable:
        add_to_histogram(histogram_dict, value)
    return histogram_dict


def normalize_histogram_dict(histogram_dict: Dict[T, int | float]) -> Dict[T, Number]:
    """
    Given a histogram (a dict mapping values to counts), returns a normalized
    histogram where each count has been divided by the sum of all counts.
    This can be seen as a distribution probability based on the initial counts.
    """
    total = sum(histogram_dict.values())
    return {value: count / total for value, count in histogram_dict.items()}


def compute_normalized_histogram_dict(iterable: Iterable[T]) -> Dict[T, Number]:
    """
    Creates a dict mapping elements in an iterable to their count divided by the total count.
    """
    histogram_dict = compute_histogram_dict(iterable)
    normalized_histogram_dict = normalize_histogram_dict(histogram_dict)
    return normalized_histogram_dict


def get_empirical_distribution(factor, number_of_samples):
    samples = [tuple(factor.sample().tolist()) for _ in range(number_of_samples)]
    empirical_distribution = compute_normalized_histogram_dict(samples)
    return empirical_distribution


@dataclass
class Normal:
    mean: float
    std: float

    def contains_within_z_score(self, value, z_score=4.75):
        """
        Indicates whether value is within the mean-centered interval with given z_score
        (default 4.75 corresponding to probability 99.99%).
        """
        return abs(value - self.mean) < z_score * self.std


def compute_sample_probability_distributions_dict(values: Iterable[T],
                                                  probability: Callable[[T], float],
                                                  number_of_samples: int) -> Dict[T, Normal]:
    """
    Returns a dict mapping each value in a multinomial distribution
    to the Normal distribution of empirical probability estimated from a number of samples,
    according to the Law of Large Numbers.
    """
    sqrt_n = math.sqrt(number_of_samples)
    sample_probability_distributions_dict = {}
    for sample in values:
        p = probability(sample)
        stddev = math.sqrt(p * (1 - p))
        stderr = stddev / sqrt_n
        sample_probability_distributions_dict[sample] = Normal(mean=p, std=stderr)
    return sample_probability_distributions_dict
