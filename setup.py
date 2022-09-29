# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# SPDX-License-Identifier: MIT

import sys

from setuptools import find_packages, setup

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10


INSTALL_REQUIRES = [
    "networkx",
    "torch",
    "torchaudio",
    "torchvision",
    "matplotlib",
    "tqdm",
    "pytest",
    "pytest-cov",
    "sympy",
    "z3-solver",
]
TEST_REQUIRES = [
    "pytest",
    "pytest-cov",
]
DEV_REQUIRES = TEST_REQUIRES + [
    "black",
    "flake8",
    "flake8-bugbear",
    "mypy",
    "usort",
    "ufmt",
]


# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)


# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="neuralpp",
    version="0.1.1",
    description="Neural Probabilistic Programs (NeuralPPs)",
    author="Rodrigo de Salvo Braz",
    license="MIT",
    url="https://github.com/rodrigodesalvobraz/neuralpp",
    project_urls={
        "Documentation": "https://github.com/rodrigodesalvobraz/neuralpp",
        "Source": "https://github.com/rodrigodesalvobraz/neuralpp",
    },
    keywords=[
        "Bayesian Inference",
        "Exact Inference",
        "PyTorch",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">={}.{}".format(REQUIRED_MAJOR, REQUIRED_MINOR),
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(
        include=["neuralpp", "neuralpp.*"],
        exclude=["debug", "tests", "website"],
    ),
    extras_require={
        "dev": DEV_REQUIRES,
        "test": TEST_REQUIRES,
    },
)
