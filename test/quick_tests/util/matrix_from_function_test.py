from util.util import matrix_from_function


def test_matrix_from_function():

    tests = [
        (([range(4)], lambda i: i),
         [0, 1, 2, 3]),

        (([range(2), range(3)], lambda i, j: i * 3 + j),
         [[0, 1, 2], [3, 4, 5]]),

        (([range(2), range(3), range(2)], lambda i, j, k: i * 6 + j * 2 + k),
         [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]),

        (([], lambda: 0),
         0),

        (([range(0)], lambda i: i),
         []),

        (([range(0), range(3)], lambda i, j: i * 3 + j),
         []),

        (([range(2), range(0), range(2)], lambda i, j, k: i * 6 + j * 2 + k),
         [[], []]),
    ]

    for test in tests:
        assert matrix_from_function(*test[0]) == test[1]
