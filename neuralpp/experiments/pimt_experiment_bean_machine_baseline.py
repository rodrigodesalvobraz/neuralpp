import time

import torch
from beanmachine.ppl.inference import BMGInference
from torch import distributions as dist
import beanmachine.ppl as bm

# torch.manual_seed(-5)


@bm.random_variable
def mu2():
    return dist.Normal(0, 1)


@bm.random_variable
def mu1():
    return dist.Normal(mu2(), 1)


@bm.random_variable
def x():
    return dist.Normal(mu1(), 1)


for inference in [bm.inference.BMGInference(), bm.CompositionalInference()]:
    start = time.time()
    samples = inference.infer(
        queries=[mu2()],
        observations={x(): torch.tensor(0.0)},
        num_samples=5_000,
        num_chains=1,
    )
    end = time.time()
    print(f"Inference {inference} found {samples[mu2()][0].mean()} in {(end - start):.3} secs.")