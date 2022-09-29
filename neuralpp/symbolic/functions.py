""" A library that contains operators that python does not have natively. """


def conditional(if_, then_, else_):
    """a conditional operator"""
    return then_ if if_ else else_


def identity(a):
    return a
