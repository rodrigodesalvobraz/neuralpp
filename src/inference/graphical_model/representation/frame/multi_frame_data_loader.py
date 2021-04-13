from inference.graphical_model.representation.frame.tuple_data_loader import get_data_loader
from inference.graphical_model.variable.discrete_variable import DiscreteVariable


class MultiFrameDataLoader:

    def __init__(self, dataset, batch_size=100):
        try:
            len(dataset[0]) == 2 and \
            all(all(isinstance(v, DiscreteVariable) for v in dataset[0][i].keys()) for i in {0, 1})
        except TypeError as e:
            raise Exception(
                "A dataset must be a sequence of tuples of two dicts. The first dict is the observation, "
                "mapping variables to their observed values. The second dict is the query label, mapping query "
                "variables to their queried values") from e

        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return self.multi_frame_data_loader()

    def multi_frame_data_loader(self):
        # We unusually have two loops over the data: one for pairs (observation_frame, query_assignment_frame)
        # and another for the possibly multiple values associated to random variables in each of those
        # two maps.
        # (For those familiar with R or pandas, it is useful to think of both observation_frame
        # or query_assignment_frame as data frames.)
        # This happens here because we can train the same effective_model from frames defined on different
        # observed and queried variables.
        # The dataset is composed of frames, each defined on the same variables.
        # The usual training is therefore performed for each frame.
        for (observation_frame, query_assignment_frame) in self.dataset:
            data_loader = get_data_loader(observation_frame, query_assignment_frame, self.batch_size)
            for (observation_batch, query_assignment_batch) in data_loader:
                yield observation_batch, query_assignment_batch