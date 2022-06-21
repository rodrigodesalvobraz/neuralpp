from sympy.core.parameters import global_parameters as gp
from sympy.core.cache import clear_cache


class SymPyNoEvaluation:
    def __init__(self):
        pass

    def __enter__(self):
        clear_cache()
        gp.evaluate = False
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        clear_cache()
        gp.evaluate = True
        if not exc_type:
            return True
        else:
            return False  # do not suppress the raised error


