from collections import OrderedDict
from typing import Optional

NO_CACHED_VALUE = object()


def cache_by_id(fnc):
    return lru_cache_by_id(None)(fnc)


def lru_cache_by_id(capacity: Optional[int]):
    def cache_by_id_inner(fnc):
        cache = OrderedDict()

        def cached_fnc(*args):
            ids = tuple(id(arg) + hash(str(arg)) for arg in args)
            result_from_cache = cache.get(ids, NO_CACHED_VALUE)
            if result_from_cache is NO_CACHED_VALUE:
                result = fnc(*args)
                cache[ids] = result
                cache.move_to_end(ids)
                return result
            cache.move_to_end(ids)
            if capacity is not None and len(cache) > capacity:
                cache.popitem(last=False)
            return result_from_cache

        return cached_fnc

    return cache_by_id_inner
