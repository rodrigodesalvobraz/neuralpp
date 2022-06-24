"""
Test of Z3Py. Most parts are covered in `https://ericpony.github.io/z3py-tutorial/guide-examples.htm`.
"""
import pytest
import z3.z3types
from z3 import Solver, Not, sat, unsat, Int, Implies, Or, simplify, Ints, And, Reals, BitVecVal, Function, IntSort, \
    ForAll, Exists, ExprRef, Sum, Context, Array, BoolSort
from copy import copy


def is_valid(predicate: ExprRef) -> bool:
    """
    If we want to ask Z3 to check if a predicate PRED is valid (i.e., always true), we need to
    ask in a not very intuitive way: "is `not PRED` unsatisfiable?"
    If `not PRED` is sat, then PRED is NOT always true, since there must exist a counter-example that satisfies
    `not PRED`, which we can get by calling s.model().
    If `not PRED` is unsat, then PRED is always true.
    """
    s = Solver()
    s.add(Not(predicate))
    return s.check() == unsat


def test_is_valid():
    """ Test that is_valid() is implemented correctly. """
    x = Int('x')
    assert is_valid(Or(x > 0, x <= 0))
    assert is_valid(Implies(x > 1, x > 0))
    assert not is_valid(x > 0)
    assert not is_valid(Implies(x > 0, x > 1))


def test_z3_simplification():
    """ Test of z3's simplify() function. """
    x, y = Ints('x y')

    # simplify() can perform some trivial simplification.
    # == and != are overloaded. So we just use repr().
    assert repr(simplify(x < y + x + 2)) == "Not(y <= -2)"

    # We start from two equivalent condition. Ideally we want to derive cond2 automatically from cond1.
    cond1 = And(x > 2, x < 4)
    cond2 = x == 3
    # However, Z3 cannot simplify 2 < x < 4 into x == 3
    assert repr(simplify(cond1)) == "And(Not(x <= 2), Not(4 <= x))"
    # But we can always ask z3 to solve the problem "is 2 < x < 4 equivalent to x == 3?"
    assert is_valid(cond1 == cond2)


def test_z3_solve_nonlinear_polynomial():
    """
    Z3 can solve nonlinear polynomial constraints,although to a lesser extent than SymPy which
    also supports powers, exp/log and trigonometric.
    """
    x, y = Reals('x y')
    s = Solver()
    s.add(x ** 2 + y ** 2 > 3, x ** 3 + y < 5)
    assert s.check() == sat
    # one can call always s.model() to get a solution if check() == sat


def test_bitvec():
    """ BitVec is Z3's term for bit-vector. For example, 16-bit integer is a bit-vector. """
    a = BitVecVal(-1, 16)
    b = BitVecVal(65535, 16)
    assert is_valid(a == b)  # -1 (signed) is 65535 (unsigned) in 16-bit representation.
    a = BitVecVal(-1, 32)
    b = BitVecVal(65535, 32)
    assert is_valid(a != b)  # -1 is not 65535 in 32-bit representation.


def test_function():
    """ In Z3, functions are uninterpreted and total. Uninterpreted means it's just a name, we cannot give it
    any definition or interpretation; total means it has no side effects, like functions in functional language.
    """
    x = Int('x')
    f = Function('f', IntSort(), IntSort())
    assert is_valid(Implies(f(x) == x, f(f(f(x))) == x))


def test_quantifier():
    """ Z3 also supports quantifiers, such as `forall`, `exists`. """
    x, y = Ints('x y')
    f = Function('f', IntSort(), IntSort(), IntSort())
    assert is_valid(Implies(ForAll([x, y], f(x, y) >= x), f(0, 5) >= 0))  # if forall x y, f(x,y)>=x, then f(0,5)>=0
    assert is_valid(Implies(Exists([x], f(x, y) == x), Not(ForAll([x], f(x, y) != x))))  # exists


def test_sum():
    """
    `Sum` is not a quantifier in Z3, it's just an interpreted function.
    """
    # simple usage of Sum()
    x, i = Ints('x i')
    assert is_valid(Sum([1, 2, x]) == x + 3)
    # Now, say we want to check the theorem of Gauss Summation: 1 + 2 + ... n = (1 + n) * n / 2,
    # we can only do this when n is static to Z3.
    # We cannot state something like "forall n, 1 + 2 + ... n = (1 + n) * n / 2"
    N = 1000
    assert (is_valid(Sum([j for j in range(N+1)]) == (1 + N)*N/2))
    assert (is_valid(Sum([x for _ in range(N+1)]) == (1 + N)*x))
    assert (is_valid(Sum([x + j for j in range(N+1)]) == (1 + N)*x + (1 + N)*N/2))


def test_z3_solver():
    """
    We want to test if Z3's solver environment can be copied. The use case:
    say we already have constraint C, and we see a new literal L; we want to check the satisfiability of both
    "C and L" and "C and not L"; if we can clone the Z3 environment/solver, then we can treat it as immutable.
    Otherwise, we must treat it as mutable and can only check "C and L", call pop(), and then check "C and not L".
    """
    # the use case of using only one Solver()
    s = Solver()
    x, y = Ints('x y')
    constraints = And(x > 2, y < 0)
    s.add(constraints)
    literal = x > y

    # s.push() creates a new scope, so z3 knows where it should pop() to. E.g.:
    #
    #      constraints0,     <- s.add(constraints0)
    #      constraints1,     <- s.add(constraints1)
    # -------new scope------ <- s.push()
    #      constraints2,     <- s.add(constraints2)
    #      constraints3,     <- s.add(constraints3)
    #
    #  now if we call s.pop(), constraints2 and constraints3 will be deleted.
    s.push()
    s.add(x > y)
    assert s.check() == sat
    s.pop()
    s.add(Not(x > y))
    assert s.check() == unsat

    # z3 has "context", or state/environment.
    c1 = Context()
    s = Solver(ctx=c1)
    x, y = Ints('x y', ctx=c1)
    constraints = And(x > 2, y < 0)
    s.add(constraints)
    c2 = Context()
    s2 = s.translate(c2)  # "translate" s with the context c2, create a new solver object.
    s.add(Not(x > y))
    assert s.check() == unsat
    # we cannot reuse x and y in another context
    with pytest.raises(z3.z3types.Z3Exception):
        s2.add(Not(x > y))
    x2, y2 = Ints('x y', ctx=c2)
    s2.add(Not(x2 > y2))
    assert s2.check() == unsat

    # We can just operate in one context.
    # Actually, according to document of Z3Py:
    # "Z3Py uses a default global context. For most applications this is sufficient."
    c = Context()
    s = Solver(ctx=c)
    x, y = Ints('x y', ctx=c)
    constraints = And(x > 2, y < 0)
    s.add(constraints)
    s2 = s.translate(c)  # "translate" with the same context is just copy
    s.add(Not(x > y))
    assert s.check() == unsat
    # now the state of s is already unsat, if "s2 = s" is not a deep copy, s2.check() should be unsat
    s2.add(x > y)
    assert s2.check() == sat

    # Solver can take care of copying for us, we don't need to worry about context. From z3.Solver's source code:
    #   def __copy__(self):
    #       return self.translate(self.ctx)
    #
    #   def __deepcopy__(self, memo={}):
    #       return self.translate(self.ctx)
    # So last section of code above can be simplified to the following, where we don't need an explicit context.
    s = Solver()
    x, y = Ints('x y')
    constraints = And(x > 2, y < 0)
    s.add(constraints)
    s2 = copy(s)  # just use copy instead of "context"
    s.add(Not(x > y))
    assert s.check() == unsat
    s2.add(x > y)
    assert s2.check() == sat

    # merging two solvers
    x, y = Ints('x y')
    constraints = And(x > 2, y < 0)
    s1 = Solver()
    s1.add(constraints)
    s2 = Solver()
    s2.add(Not(x > y))
    s3 = copy(s1)
    assert s3.check() == sat
    s3.append(s2.assertions())
    assert s3.check() == unsat


def test_z3_sort():
    x = Int('x')
    assert x.sort() == IntSort()
    array = Array('a', IntSort(), IntSort())
    assert array.sort() == z3.ArraySort(IntSort(), IntSort())
    function = Function('f', IntSort(), IntSort(), BoolSort())
    with pytest.raises(AttributeError):
        function.sort()  # function has no attribute 'sort'

    add = x + 1  # function application has sort (which is return type)
    assert add.sort() == IntSort()
    assert function(x, x).sort() == BoolSort()
    # z3 function cannot be partially applied
    with pytest.raises(z3.Z3Exception):
        function(x)

    # to get the "type" of a function
    assert function.arity() == 2
    assert function.domain(0) == IntSort()  # args[0]
    assert function.domain(1) == IntSort()  # args[1]
    assert function.range() == BoolSort()  # return type
    with pytest.raises(z3.Z3Exception):
        function.domain(2)

    assert isinstance(function, z3.FuncDeclRef)
    assert isinstance(function(x, x), z3.ExprRef)
    with pytest.raises(TypeError):
        # z3.FPSort() requires two arguments ebits and sbits (Single = FPSort(8, 24), Double = FPSort(11, 53))
        z3.FPSort()
    assert z3.FPVal(1.33).sort() == z3.FPSort(11, 53)  # double
    assert isinstance(z3.FPVal(1.33).sort(), z3.FPSortRef)

    # a bit weird behavior from z3's arity() call
    assert z3.And(True, True).decl().arity() == 2
    assert z3.And(True, False, True).decl().arity() == 2
    assert z3.And(True, True, True, True).decl().arity() == 2
    assert z3.And(True, True, True, True).decl() == z3.And(True, True).decl()

    # but it does distinguish an int add and a real add.
    y = z3.Real("y")
    assert (y+y).decl().kind() == (x+x).decl().kind()
    assert (y+y).decl() != (x+x).decl()
    assert (y+y).decl().domain(0) == z3.RealSort()
    assert (x+x).decl().domain(0) == z3.IntSort()


def test_z3_fp_sort():
    for sort in [z3.RealSort(), z3.IntSort()]:
        x, y = z3.Consts("x y", sort)
        fp_add = (x + y).decl()
        assert fp_add.arity() == 2
        assert fp_add.domain(0) == sort
        assert fp_add.domain(1) == sort
        assert fp_add.range() == sort

    # fp in z3 is a bit more complicated..
    double_sort = z3.FPSort(11, 53)
    x, y = z3.Consts("x y", double_sort)
    fp_add = (x + y).decl()
    assert fp_add.arity() == 3  # instead of 2
    assert isinstance(fp_add.domain(0), z3.FPRMSortRef)
    assert fp_add.domain(1) == double_sort
    assert fp_add.domain(2) == double_sort
    assert fp_add.range() == double_sort
