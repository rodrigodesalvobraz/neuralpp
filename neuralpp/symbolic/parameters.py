"""Thread-safe global parameters. This file mimics sympy.core.parameters. """

from contextlib import contextmanager
from threading import local


class _GlobalParameters(local):
    """
    Thread-local global parameters.

    Explanation
    ===========

    This class generates thread-local container for SymPy's global parameters.
    Every global parameters must be passed as keyword argument when generating
    its instance.
    A variable, `global_parameters` is provided as default instance for this class.

    References
    ==========

    .. [1] https://docs.python.org/3/library/threading.html
    .. [2] sympy.core.parameters._global_parameters

    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setattr__(self, name, value):
        return super().__setattr__(name, value)


global_parameters = _GlobalParameters(sympy_evaluate=False)


@contextmanager
def sympy_evaluate(x):
    old = global_parameters.sympy_evaluate
    try:
        global_parameters.sympy_evaluate = x
        yield
    finally:
        global_parameters.sympy_evaluate = old
