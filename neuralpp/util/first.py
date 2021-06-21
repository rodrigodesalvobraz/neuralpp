def first(iterable, condition=lambda x: True, default=None):
    """
    Returns the first item in the `iterable` that
    satisfies the `condition`.

    If the condition is not given, returns the first item of
    the iterable.

    If there is no first element, the default is returned (None if default is not provided).
    """

    try:
        return next(x for x in iterable if condition(x))
    except StopIteration:
        return default
