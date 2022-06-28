""" A library that contains operators that python does not have natively. """


def cond(if_, then_, else_):
    """ a conditional operator """
    return then_ if if_ else else_
