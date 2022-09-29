from neuralpp.inference.graphical_model.representation.frame.dict_data_loader import (
    DictDataLoader,
)
from neuralpp.inference.graphical_model.representation.frame.dict_frame import (
    generalized_len_of_dict_frame,
    is_frame,
)


class TupleDataLoader:
    def __init__(self, dataset_tuple, component_data_loader_maker):
        self.dataset_tuple = dataset_tuple
        self.data_loaders = [
            component_data_loader_maker(ds) for ds in self.dataset_tuple
        ]
        lengths = {len(data_loader) for data_loader in self.data_loaders}
        if len(lengths) != 1:
            raise DataLoadersMustHaveTheSameLength()
        (self.length,) = lengths

    def __iter__(self):
        return zip(*self.data_loaders)

    def __len__(self):
        return self.length


class DataLoadersMustHaveTheSameLength(BaseException):
    def __init__(self):
        super(DataLoadersMustHaveTheSameLength, self).__init__(
            "Data loaders must have the same length"
        )


def get_data_loader(
    observation_frame, query_assignment_frame, batch_size=100
):
    try_breaking_data_into_batches = True
    dictionary = {**observation_frame, **query_assignment_frame}
    use_batches = try_breaking_data_into_batches and is_frame(dictionary)
    if use_batches:
        number_of_values = generalized_len_of_dict_frame(dictionary)
        batch_size = (
            batch_size if number_of_values > batch_size else number_of_values
        )
        frame_data_loader_maker = lambda frame: DictDataLoader(
            frame, batch_size=batch_size
        )
        data_loader = TupleDataLoader(
            (observation_frame, query_assignment_frame),
            frame_data_loader_maker,
        )
    else:
        data_loader = [(observation_frame, query_assignment_frame)]
    return data_loader
