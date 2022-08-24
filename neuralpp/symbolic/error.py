from typing import Callable


class AmbiguousTypeError(TypeError, ValueError):
    def __init__(self, python_callable: Callable):
        super().__init__(f"{python_callable} is ambiguous.")


class ConversionError(Exception):
    pass


class NotTypedError(ValueError, TypeError):
    pass


class FunctionNotTypedError(NotTypedError):
    pass


class VariableNotTypedError(NotTypedError):
    pass


class UnknownError(ValueError, RuntimeError):
    pass
