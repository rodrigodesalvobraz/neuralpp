# Neural Probabilistic Programs (NeuralPPs)

This is an implementation of graphical models inference (Variable Elimination) using PyTorch for implementing
factors (conditional probabilities and priors).
Those can be implemented as tables (the typical discrete graphical model implementation) but also as arbitrary
PyTorch modules, the case of most interest being of course neural networks.

The reason we are calling it "probabilistic programs" even though we only have discrete graphical models so far is that factors and variables are implemented in a manner general enough for probabilistic programming symbolic implementations be introduced down the road.

## Purpose

This project is for researchers interested in exploring the connection between probabilistic and neural network inference.
It is a library that provides the utilities for mixing neural networks and graphical models while offering GPU use and stochastic gradient descent training.

# Content

This repository contains:

* source code, including an experiments directory
* test code (to be run with PyTest)
* `setup.py` for installing with Pip
* `environment.yml` for importing a Conda environment 

## Installation

### PyPI

The library can be installed with

```
pip install neuralpp
```

### Installation from source

First, clone the repo locally:

```
git clone https://github.com/rodrigodesalvobraz/neuralpp.git
cd neuralpp
```

Then, to install a developer copy, run:

```
python setup.py develop
```

Alternatively, for a regular installation:

```
python setup.py install
```

## Experiments

In `src/experiments` one can find `src/experiments/simple_mnist.py`, which shows how to use a "graphical model" with a single factor,
implemented by a convolutional neural network, to learn how to recognize MNIST digits.

In `src/experiments/successive_digits.py` 
there is code for training a MNIST recognizer from pairs of images of digits, 
labeled as being successive digits (positive examples) or not (negative examples).
The correct digit labels are still learned in spite of a total absence of digit labels.
This is possible due to the reasoning performed by the graphical model component of the model
(based on the knowledge of what successive digits are).

In `src/experiments/sum_of_pair.py` pairs of images are labeled by the sum of their corresponding digits.
Again the reasoning aspect of graphical models helps by using knowledge about addition.

## Tests

Tests in `neuralpp.test` are split into `quick_tests` and `slow_tests`.
The former include tests of basic data structure implementations while the latter includes  learning sessions with stochastic gradient descent and take several minutes.

Run them with `pytest .` from the root directory if installed from the source code.