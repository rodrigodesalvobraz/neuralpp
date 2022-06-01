from neuralpp.util.util import list_for_each


def test_list_for_each():
    actual = list_for_each(range(1,5), lambda i: i)
    expected = [1, 2, 3, 4]
    assert actual == expected

    pre_list = []
    actual = list_for_each(range(1,5), lambda i: pre_list.append(i), lambda i: i)
    expected = [1, 2, 3, 4]
    assert pre_list == expected
    assert actual == expected

    actual = list_for_each(range(1,5), lambda i: i, filter_index=lambda i: i != 3)
    expected = [1, 2, 4]
    assert actual == expected

    pre_list = []
    actual = list_for_each(range(1,5), lambda i: pre_list.append(i), lambda i: i, filter_index=lambda i: i != 3)
    expected = [1, 2, 4]
    assert pre_list == [1, 2, 3, 4]
    assert actual == expected

    pre_list = []
    actual = list_for_each(range(1,5), lambda i: pre_list.append(i), lambda i: 2*i, filter_element=lambda i: i != 6)
    expected = [2, 4, 8]
    assert pre_list == [1, 2, 3, 4]
    assert actual == expected

    pre_list = []
    actual = list_for_each(range(1,5), lambda i: pre_list.append(i), lambda i: 2*i, post=lambda result: print(result))
    expected = [2, 4, 6, 8]
    assert pre_list == [1, 2, 3, 4]
    assert actual == expected

    pre_list = []
    actual = list_for_each(range(1,5), lambda i: pre_list.append(i), lambda i: 2*i,
                           post_index_result=lambda index, result: print(index, result))
    expected = [2, 4, 6, 8]
    assert pre_list == [1, 2, 3, 4]
    assert actual == expected
