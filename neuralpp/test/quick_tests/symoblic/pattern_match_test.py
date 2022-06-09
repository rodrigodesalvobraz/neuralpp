"""Testing the pattern match feature of Python which I'm not very familiar with."""


# this example is from https://peps.python.org/pep-0636/
import builtins
import operator


class Click:
    __match_args__ = ("position", "button")

    def __init__(self, pos, btn):
        self.position = pos
        self.button = btn


def test_pattern_matching():
    e = Click(pos=(1, 1), btn="left")
    match e:
        case Click(position=(x, y), button="left"):
            assert x == 1 and y == 1
        case _:
            assert False  # not reachable


# The above example is ambiguous:
# is __match_args__ associated with the argument (pos, btn),
# or the attributes (self.position, self.button)?
# In the following I switch the two elements in __match_args__
# to find out
class Click2:
    __match_args__ = ("button", "position")

    def __init__(self, pos, btn):
        self.position = pos
        self.button = btn


def test_pattern_matching_clarification():
    e = Click2(pos=(1, 1), btn="left")
    match e:
        # if ("button", "position") refers to args, then "button"=pos "position"=btn
        case Click2(position="left", button=(x, y)):
            assert False  # not reachable, so not the case
        # or, if ("button", "position") refers to attributes, then "button"=self.button, "position"=self.position
        case Click2(position=(x, y), button="left"):
            assert x == 1 and y == 1


# The following example shows that __match_args__ can be properties.
class Click3:
    __match_args__ = ("button", "position")

    def __init__(self, pos, btn):
        self._inner_position = pos
        self._inner_button = btn

    @property
    def position(self):
        return self._inner_position

    @property
    def button(self):
        return self._inner_button


def test_pattern_matching_property():
    e = Click3(pos=(1, 1), btn="left")
    match e:
        case Click3(position="left", button=(x, y)):
            assert False  # not reachable
        case Click3(position=(x, y), button="left"):  # property is also accepted as __match_args__
            assert x == 1 and y == 1


# But what if Click4 does not have __init__?
class Click4:
    __match_args__ = ("button", "position")

    @property
    def position(self):
        return self._inner_position

    @property
    def button(self):
        return self._inner_button

    @position.setter
    def position(self, pos):
        self._inner_position = pos

    @button.setter
    def button(self, btn):
        self._inner_button = btn


def test_pattern_matching_no_init():
    e = Click4()
    e.position = (1, 1)
    e.button = "left"
    match e:
        case Click4(position="left", button=(x, y)):
            assert False  # not reachable
        case Click4(position=(x, y), button="left"):  # property is also accepted as __match_args__
            assert x == 1 and y == 1


# See if pattern matching recognize python operator
class OperatorTestClass:
    __match_args__ = ("op",)

    def __init__(self, op):
        self._operator = op

    @property
    def op(self):
        return self._operator


def test_operator_matching():
    e = OperatorTestClass(operator.add)
    e2 = OperatorTestClass(operator.add)
    assert operator.add == operator.add
    assert e.op == operator.add  # a bit counterintuitive here
    assert e.op == e.op
    assert e.op == e2.op

    match e:
        case OperatorTestClass(op=operator.add):
            assert True  # not reachable
        case OperatorTestClass(op=op):  # property is also accepted as __match_args__
            assert False
