import os
import pickle

from neuralpp.util.util import check_path_like


def pickle_cache(thunk, path, refresh=False):
    if not refresh:
        check_path_like(path, caller="pickle_cache")
        try:
            with open(path, "rb") as file:
                print(f"Reading from {path}...")
                result = pickle.load(file)
                print(f"Finished reading from {path}...")
                return result
        except FileNotFoundError:
            pass

    print(f"Computing from scratch...")
    value = thunk()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        print(f"Saving to {path}...")
        pickle.dump(value, file)
    return value
