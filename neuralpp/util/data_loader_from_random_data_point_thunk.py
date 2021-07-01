class DataLoaderFromEpochGeneratorThunk:
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()


def random_epoch_generator(epoch_size, random_data_point_thunk, print=print):
    for batch_number in range(epoch_size):
        if print is not None and batch_number % 50 == 0:
            print(f"Batch: {batch_number}/{epoch_size}")
        data_point = random_data_point_thunk()
        yield data_point


def data_loader_from_random_data_point_generator(
    epoch_size, random_data_point_generator, print=print
):
    return DataLoaderFromEpochGeneratorThunk(
        lambda: random_epoch_generator(
            epoch_size, random_data_point_generator, print=print
        )
    )
