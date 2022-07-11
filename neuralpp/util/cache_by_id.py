NO_CACHED_VALUE = object()


def cache_by_id(fnc):
    cache = {}

    def cached_fnc(*args):
        ids = tuple(id(arg) for arg in args)
        result_from_cache = cache.get(ids, NO_CACHED_VALUE)
        if result_from_cache is NO_CACHED_VALUE:
            result = fnc(*args)
            cache[ids] = result
            return result
        return result_from_cache

    return cached_fnc
