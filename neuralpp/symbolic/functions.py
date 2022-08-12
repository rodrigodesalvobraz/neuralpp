""" A library that contains operators that python does not have natively. """


def conditional(if_, then_, else_):
    """ a conditional operator """
    return then_ if if_ else else_


def identity(a):
    return a


def piecewise(expression_condition_pairs):
    """
    @param expression_condition_pairs: like SymPy's Piecewise: [(expr_1, cond_1), .., (expr_n, cond_n)].
            Note cond_1 .. n should be mutually exclusive
    @return: a piecewise function
    """
    raise NotImplementedError("this is a placeholder, not supposed to be called!")
