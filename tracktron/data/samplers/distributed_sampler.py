# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from detectron2.utils import comm


class VideoTrainingSampler(Sampler):
    """
    For each sampled index, we randomly select another index from the
    `pair_indices` according to its dataset record.

    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, dataset, sampler_pair_offsets, sampler_cur_frame_weight=0,
                 num_frames=2, shuffle=True, seed=None):
        """
        Args:
            dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
                or a map-style pytorch dataset.
            sampler_pair_offsets (iterable[int]): frame offsets to be sampled.
            sampler_cur_frame_weight (float): sampling weight of current frame.
                Other frames have equal weight 1.
            num_frames (int): number of sampled frames in a group.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = len(dataset)
        assert len(dataset) > 0
        self._dataset = dataset

        self._sampler_pair_offsets = sampler_pair_offsets
        self._sampler_cur_frame_weight = sampler_cur_frame_weight
        self._sampler_pair_prob = torch.ones(len(sampler_pair_offsets), dtype=torch.float32)
        if 0 in sampler_pair_offsets:
            self._sampler_pair_prob[sampler_pair_offsets.index(0)] = sampler_cur_frame_weight
        self._num_frames = num_frames
        assert len(sampler_pair_offsets) >= num_frames >= 2

        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), self._rank, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            epoch_indices = torch.randperm(self._size, generator=g) \
                if self._shuffle else torch.arange(self._size)
            valid_pair_indices = torch.tensor([
                [self._dataset[index]["pair_indices"][pair_offset]
                 for pair_offset in self._sampler_pair_offsets] for index in epoch_indices
            ])
            epoch_pair_offsets_indices = torch.multinomial(
                self._sampler_pair_prob.repeat(self._size, 1) * (valid_pair_indices >= 0).float(),
                self._num_frames - 1, replacement=False, generator=g
            )
            epoch_pair_offsets = torch.tensor(self._sampler_pair_offsets)[epoch_pair_offsets_indices]
            epoch_pair_indices = torch.tensor([
                [self._dataset[index]["pair_indices"][pair_offset] for pair_offset in pair_offsets]
                for index, pair_offsets in zip(epoch_indices, epoch_pair_offsets)
            ])
            yield from torch.hstack((epoch_indices.unsqueeze(-1), epoch_pair_indices))


class VideoInferenceSampler(Sampler):
    """
    Produce indices for video inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of videos is not divisible by the number of workers,
    this sampler produces different number of videos on different workers.
    """

    def __init__(self, size, first_frame_indices):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            first_frame_indices (list): first frame index of each video
        """
        self._size = size
        self._first_frame_indices = first_frame_indices
        assert size > 0 and len(self._first_frame_indices) > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        frame_indices = np.array(self._first_frame_indices + [self._size])
        begin_index = int(np.argmin(np.abs(frame_indices - self._size * self._rank / self._world_size)))
        end_index = int(np.argmin(np.abs(frame_indices - self._size * (self._rank + 1) / self._world_size)))
        self._local_indices = range(int(frame_indices[begin_index]), int(frame_indices[end_index]))

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
