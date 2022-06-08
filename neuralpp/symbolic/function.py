from typing import Callable
from abc import ABC, abstractmethod


class Function(ABC):
    """
    The motivation of having a dedicated Function class is to support `built-in functions`.
    Only having Python callable is fine when dealing with BasicFunctionApplication, such as
        BasicFunctionApplication(BasicConstant(lambda x, y: x + y), [BasicConstant(1), BasicConstant(1)]).

    But when we are asked to initialize a SymPyFunctionApplication like the following:
        SymPyFunctionApplication(BasicConstant(lambda x, y: x + y), [BasicConstant(1), BasicConstant(1)]),
    we have to raise an error as we have no way to know that "lambda x, y: x + y" is actually an "add".

    This can be solved by defining a Function class.
    If we initialize the object like
        SymPyFunctionApplication(BasicConstant(Add()), [BasicConstant(1), BasicConstant(1)])
    where Add() is an subclass of Function(), we are able to recognize that Add() is an "add", and thus choose the
    underlying SymPy representation for "add" correctly.
    """
    @property
    @abstractmethod
    def arity(self) -> int:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def lambdify(self) -> Callable:
        pass


class Add(Function):
    @property
    def arity(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "add"

    def lambdify(self) -> Callable:
        return lambda x, y: x + y
