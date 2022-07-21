from neuralpp.util.cache_by_id import cache_by_id


class NumWrapper:
    """
    Test class to ensure that object ID rather than __eq__ or __hash__ is being used.
    This is used in place of ints, which are cached by Python and therefore have consistent ids.
    """

    def __init__(self, n):
        self.value = n

    def __eq__(self, other):
        return isinstance(other, NumWrapper) and other.value == self.value


num_wrappers = [NumWrapper(n) for n in range(11)]


class CountedAdd:
    """
    Counts the number of times the add function is called and executed, excluding cache retrieval.
    """
    call_count = 0

    @cache_by_id
    def add(self, x: int, y: int):
        self.call_count += 1
        return x + y


class CountedFibonacci:
    """
    Counts the number of times the fibonacci function is called and executed, excluding cache retrieval.
    """
    call_count = 0

    @cache_by_id
    def fibonacci(self, n: NumWrapper):
        self.call_count += 1
        value = n.value
        if value <= 1:
            return 1
        return self.fibonacci(num_wrappers[value-1]) + self.fibonacci(num_wrappers[value-2])


def test_cache_by_id_on_loop():
    test_object_1 = CountedAdd()
    test_object_2 = CountedAdd()
    n = 10
    m = 5

    assert (test_object_1.call_count == 0)

    for x in range(n):
        for y in range(n):
            result = test_object_1.add(x, y)
            assert (result == x + y)

    # Each call to the method should have been counted.
    assert (test_object_1.call_count == n * n)

    # Call the same results again with the same test object
    # These results should all be cached, so there are no direct calls to add
    for x in range(n):
        for y in range(n):
            test_object_1.add(x, y)

    # The count on this object is still the same as before.
    assert (test_object_1.call_count == n * n)

    # Calls to the method on test_object_2 should be stored separately, so
    # it should have m * m calls of its own even though the non-self args
    # are the same.
    assert(test_object_2.call_count == 0)
    for x in range(m):
        for y in range(m):
            test_object_2.add(x, y)
    assert (test_object_2.call_count == m * m)


def test_fibonacci_recursive():
    test_object = CountedFibonacci()

    # Check that fibonacci is computed only once per value in num_wrappers
    test_object.fibonacci(num_wrappers[9])
    assert(test_object.call_count == 10)

    # no new calls when using the same input object
    test_object.fibonacci(num_wrappers[9])
    assert (test_object.call_count == 10)

    # This next call should only result in one non-cached call
    test_object.fibonacci(num_wrappers[10])
    assert(test_object.call_count == 11)

    # this new object does result in a new call, since it is equal
    # to but not the same as the previous wrapper used for 9
    test_object.fibonacci(NumWrapper(9))
    assert(test_object.call_count == 12)

