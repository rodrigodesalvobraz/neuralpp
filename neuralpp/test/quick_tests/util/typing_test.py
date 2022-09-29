from typing import Callable, get_args


def test_callable():
    b = [bool for i in range(2)]
    c1 = Callable[
        b, int
    ]  # it's weird that PyCharm gives error on this (but it interprets)
    assert c1 == Callable[[bool, bool], int]

    c2 = Callable[[b], int]
    assert c1 != c2
    assert get_args(c1) == get_args(c2)  # also a bit unexpected

    c3 = Callable[b[5:], int]
    c4 = Callable[[b[5:]], int]
    assert get_args(c3) == get_args(c4)
