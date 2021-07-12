# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import numpy as np
import operator
import torch.utils.data
from tabulate import tabulate
from termcolor import colored

from detectron2.config import configurable
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import log_first_n
from detectron2.data.build import load_proposals_into_dataset, print_instances_class_histogram
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.detection_utils import check_metadata_consistency

from .dataset_mapper import VideoDatasetMapper
from .samplers import VideoTrainingSampler, VideoInferenceSampler
from .common import MapPairDataset, AspectRatioGroupedPairDataset

"""
This file inherits default logic from `detectron2.data.build` to build a dataloader 
for pairwise training and video sequence testing.
"""

__all__ = [
    "build_pair_batch_data_loader",
    "build_video_detection_train_loader",
    "build_video_detection_test_loader",
    "get_video_detection_dataset_dicts"
]


def print_video_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    classes = np.asarray(
        list({"{0}_{1}_{2}".format(entry["video_id"], x["category_id"], x["instance_id"]): x["category_id"]
              for entry in dataset_dicts if entry["video_id"] != -1 for x in entry["annotations"]
              if (not x.get("iscrowd", False))}.values()), dtype=np.int
    )
    if len(classes):
        assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
        assert (
            classes.max() < num_classes
        ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
    hist_bins = np.arange(num_classes + 1)
    histogram = np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of video instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )


def get_video_detection_dataset_dicts(
    dataset_names, filter_empty=True, min_keypoints=0, proposal_files=None
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        dataset_names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `dataset_names`.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None:
        assert len(dataset_names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    if "pair_indices" in dataset_dicts[0][0]:
        index_shift = 0
        for dataset_dicts_i in dataset_dicts:
            for record in dataset_dicts_i:
                record["pair_indices"] = [
                    -1 if idx < 0 else idx + index_shift for idx in record["pair_indices"]
                ]
            index_shift += len(dataset_dicts_i)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    # no filtering here for consistent indices to pair later
    # if filter_empty and has_instances:
    #     dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    # if min_keypoints > 0 and has_instances:
    #     dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if has_instances:
        try:
            class_names = MetadataCatalog.get(dataset_names[0]).thing_classes
            check_metadata_consistency("thing_classes", dataset_names)
            print_instances_class_histogram(dataset_dicts, class_names)
            print_video_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(dataset_names))
    return dataset_dicts


def build_pair_batch_data_loader(
    dataset, sampler, total_batch_size, *, aspect_ratio_grouping=False, num_workers=0
):
    """
    Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size, aspect_ratio_grouping, num_workers): see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedPairDataset(data_loader, batch_size)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last=True
        )  # drop_last so the batch always have the same size
        return ChainedDataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )


def _video_train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_video_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

    if mapper is None:
        mapper = VideoDatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "VideoTrainingSampler":
            sampler = VideoTrainingSampler(dataset, cfg.DATALOADER.SAMPLER_PAIR_OFFSETS,
                                           cfg.DATALOADER.SAMPLER_CUR_FRAME_WEIGHT,
                                           cfg.INPUT.NUM_FRAMES_TRAIN)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "sampler_pair_offsets": cfg.DATALOADER.SAMPLER_PAIR_OFFSETS,
        "sampler_cur_frame_weight": cfg.DATALOADER.SAMPLER_CUR_FRAME_WEIGHT,
        "num_frames": cfg.INPUT.NUM_FRAMES_TRAIN,
    }


# TODO can allow dataset as an iterable or IterableDataset to make this function more general
@configurable(from_config=_video_train_loader_from_config)
def build_video_detection_train_loader(
    dataset, *, mapper, sampler=None, total_batch_size,
    aspect_ratio_grouping=True, num_workers=0, sampler_pair_offsets=(-1,),
    sampler_cur_frame_weight=0, num_frames=2
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that
            produces indices to be applied on ``dataset``.
            Default to :class:`TrainingSampler`, which coordinates a random shuffle
            sequence across all workers.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers.
        sampler_pair_offsets (iterable[int]): frame offset options to pair.
        sampler_cur_frame_weight (float): sampling weight of current frame.
            Other frames have equal weight 1.
        num_frames (int): number of sampled frames in a group.

    Returns:
        torch.utils.data.DataLoader: a dataloader. Each output from it is a
            ``list[mapped_element]`` of length ``total_batch_size / num_workers``,
            where ``mapped_element`` is produced by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapPairDataset(dataset, mapper)
    if sampler is None:
        sampler = VideoTrainingSampler(
            dataset, sampler_pair_offsets, sampler_cur_frame_weight, num_frames
        )
    assert isinstance(sampler, torch.utils.data.sampler.Sampler)
    return build_pair_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
    )


def _video_test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    dataset = get_video_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(dataset_name)]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    if mapper is None:
        mapper = VideoDatasetMapper(cfg, False)
    return {
        "dataset": dataset,
        "mapper": mapper,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "first_frame_indices": MetadataCatalog.get(dataset_name).first_frame_indices,
        # "sampler_pair_offset": cfg.DATALOADER.SAMPLER_PAIR_OFFSET_TEST,
    }


@configurable(from_config=_video_test_loader_from_config)
def build_video_detection_test_loader(dataset, *, mapper, first_frame_indices, num_workers=0):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a map-style pytorch dataset. They can be obtained by using
            :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        first_frame_indices (list): first frame index of each video
        num_workers (int): number of parallel data loading workers

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    sampler = VideoInferenceSampler(len(dataset), first_frame_indices)
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    seed_all_rng(np.random.randint(2 ** 31) + worker_id)


class ChainedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super(ChainedDataLoader, self).__init__(*args, **kwargs)

    def __iter__(self):
        for data in super().__iter__():
            yield [pair_data[i] for i in range(len(data[0])) for pair_data in data]
