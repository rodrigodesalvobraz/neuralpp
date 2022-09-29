import beanmachine.ppl as bm
import arviz as az
import xarray as xr
from typing import NamedTuple, List, Dict, Optional
import time
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class BenchmarkResult(NamedTuple):
    samples: bm.inference.monte_carlo_samples.MonteCarloSamples
    sample_sizes: np.ndarray  # number of samples included in each measurement
    inference_time: np.ndarray  # amount of time spent on inference
    log_likelihood: np.ndarray  # log likelihood (of the observations)
    ess: xr.Dataset  # effective sample size
    rhat: xr.Dataset  # rhat convergence measurement


def benchmark(
    infer_class: bm.inference.base_inference.BaseInference,
    queries: List[bm.RVIdentifier],
    observations: Dict[bm.RVIdentifier, torch.Tensor],
    num_samples: int,
    num_chains: int = 4,
    num_adaptive_samples: int = 500,
    interval: int = 100,
) -> BenchmarkResult:
    if num_adaptive_samples is None:
        num_adaptive_samples = num_samples // 2

    # create a list of batch size. The cumsum of this will be the x-axis of the
    # measurements
    num_batch = num_samples // interval
    batch_sizes = [num_adaptive_samples] + [interval] * num_batch
    if num_samples % interval != 0:
        batch_sizes.append(num_samples % interval)

    all_samples = []
    all_sample_time = []
    log_likelihood = []

    # The following section runs each iteration of the inference "manually" -- we're
    # doing so here to measure the inference time against varying number of samples
    for _ in range(num_chains):
        # construct the sampler and wrap it around a progress bar
        sampler = infer_class.sampler(
            queries, observations, num_samples, num_adaptive_samples
        )

        chain_samples = {query: [] for query in queries}
        sample_time = []
        log_ll = []

        with tqdm(total=num_samples + num_adaptive_samples) as pbar:
            for n in batch_sizes:
                # draw the next n samples and measure the run time
                begin_time = time.perf_counter()
                next_n_worlds = []
                for _ in range(n):
                    next_n_worlds.append(next(sampler))  # draw next sample
                    pbar.update()  # update progress bar
                end_time = time.perf_counter()
                sample_time.append(end_time - begin_time)
                # collect samples
                for world in next_n_worlds:
                    for query in queries:
                        chain_samples[query].append(world.call(query))
                    log_ll.append(world.log_prob(observations).item())

        all_samples.append(
            {query: torch.stack(val) for query, val in chain_samples.items()}
        )
        # drop adaptive iterations
        all_sample_time.append(np.cumsum(sample_time[1:]))
        log_likelihood.append(log_ll)

    mcs = bm.inference.monte_carlo_samples.MonteCarloSamples(
        all_samples, num_adaptive_samples
    )
    sample_sizes = np.cumsum(batch_sizes[1:])
    all_sample_time = np.array(all_sample_time)

    # convert to xarray dataset, which works better with ArviZ
    xr_dataset = mcs.to_xarray()
    ess = []
    ess_per_second = []
    rhat = []
    # Note: ArviZ actually has a similar plot_ess function that displays the "evolution"
    # of effective sample size, but it doesn't allow us to compare samples draw from
    # two inferences
    for idx, n in enumerate(sample_sizes):
        first_n_samples = xr_dataset.isel(draw=slice(None, n))
        ess.append(az.ess(first_n_samples).to_array())
        ess_per_second.append(ess[-1] / np.mean(all_sample_time[:, idx]))
        rhat.append(az.rhat(first_n_samples).to_array())

    mcs = bm.inference.monte_carlo_samples.MonteCarloSamples(
        all_samples, num_adaptive_samples
    )
    sample_sizes = np.cumsum(batch_sizes[1:])
    all_sample_time = np.array(all_sample_time)

    return BenchmarkResult(
        samples=mcs,
        sample_sizes=sample_sizes,
        inference_time=all_sample_time,
        log_likelihood=np.array(log_likelihood),
        ess=xr.concat(ess, dim="draw"),
        rhat=xr.concat(rhat, dim="draw"),
    )


def plot_metric(
    metric: str, sample_sizes: np.ndarray, values: Dict[str, xr.Dataset]
):

    plt.figure()
    for infer_type, val in values.items():
        lines = plt.plot(
            sample_sizes, val.mean(dim="variable"), label=infer_type
        )
        plt.fill_between(
            sample_sizes,
            val.min(dim="variable"),
            val.max(dim="variable"),
            color=lines[-1].get_color(),
            interpolate=True,
            alpha=0.3,
        )
    plt.xlabel("sample size")
    plt.ylabel(metric)
    plt.legend()
    plt.show()


def generate_plots(benchmark_results: Dict[str, BenchmarkResult]):
    assert len(benchmark_results) > 0
    sample_sizes = next(iter(benchmark_results.values())).sample_sizes

    # ess vs sample size
    plot_metric(
        "effective sample size",
        sample_sizes,
        {
            method_name: result.ess
            for method_name, result in benchmark_results.items()
        },
    )

    # rhat vs sample size
    plot_metric(
        "R_hat",
        sample_sizes,
        {
            method_name: result.rhat
            for method_name, result in benchmark_results.items()
        },
    )
