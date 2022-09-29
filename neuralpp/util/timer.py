import time


class Timer:
    def __init__(self, *description_args):
        if description_args and type(description_args[0]) == bool:
            self.active = description_args[0]
            description_args = description_args[1:]
        else:
            self.active = True
        self.description_args = description_args

    def __enter__(self):
        if self.active:
            print("\nStarted", *self.description_args)
            self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.active:
            self.end = time.time()
            print(
                "Finished",
                *(self.description_args + (f"({self.end - self.start:.2f} seconds)",)),
            )


if __name__ == "__main__":
    with Timer("test"):
        [i for i in range(100000)]
    with Timer("test2"):
        [i for i in range(100000)]
    with Timer(True, "This should appear"):
        [i for i in range(100000)]
    with Timer(False, "This should not appear!"):
        [i for i in range(100000)]
