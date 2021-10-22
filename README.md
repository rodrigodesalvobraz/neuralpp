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
* `environment.yml` for importing a Conda environment 

## Experiments

In `src/experiments` one can find `src/experiments/simple_mnist.py`, which shows how to use a "graphical model" with a single factor,
implemented by a convolutional neural network, to learn how to recognize MNIST digits.

In `src/experiments/mnist_pairs_semi_supervised.py` there is code attempting to train a MNIST recognizer from pairs of images of successive digits only. It does not yet work but the script offers many options for running simplified versions of the problem, many of which do work.

## Tests

Tests in `neuralpp.test` are split into `quick_tests` and `slow_tests`.
The former include tests of basic data structure implementations while the latter includes  learning sessions with stochastic gradient descent and take several minutes.

Run them with `pytest .` from the root directory.