# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import (
    StandardROIHeads, ROI_HEADS_REGISTRY, add_ground_truth_to_proposals
)

from .embed_head import build_embed_head

logger = logging.getLogger(__name__)


def sample_proposals_for_embed_head(
    proposals: List[Instances], num_samples: int, positive_fraction: float,
    upper_neg_pos_ratio: float = 3, pos_iou_thresh: float = 0.7,
    neg_iou_thresh: float = 0.3, num_bins: int = 3,
) -> List[Instances]:
    """
    Given a list of N Instances (for N images), each containing a `gt_ious` field,
    sample with a simplified strategy as `MaxIoUAssigner` and `IoUBalancedNegSampler`
    implemented in `mmdetection` framework.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        num_samples (int): The total number of proposals to return.
        positive_fraction (float): The number of positive proposals with `gt_ious`
            >= pos_iou_thresh is `min(num_positives, int(positive_fraction * num_samples))`.
            The number of negatives samples with `gt_ious` < neg_iou_thresh is
            `min(num_negatives, num_samples - num_positives_sampled,
            num_positives_sampled * upper_neg_pos_ratio)`.
        upper_neg_pos_ratio (float): Upper bound ratio of negative and positive samples.
        pos_iou_thresh (float): IoU threshold for positive proposals.
        neg_iou_thresh (float): IoU threshold for negative proposals.
        num_bins (int): Number of bins for IoU balanced negative sampling.

    Returns:
        list[Instances]: Filtered instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_ious")
    assert num_bins > 0

    filtered_proposals = []
    for proposals_per_image in proposals:
        positive = nonzero_tuple(proposals_per_image.gt_ious >= pos_iou_thresh)[0]
        negative = nonzero_tuple(proposals_per_image.gt_ious < neg_iou_thresh)[0]
        num_pos = min(positive.numel(), int(num_samples * positive_fraction))
        num_neg = min(negative.numel(), num_samples - num_pos, num_pos * upper_neg_pos_ratio)

        if positive.numel() > num_pos:
            selected_pos = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            positive = positive[selected_pos]
        if negative.numel() > num_neg:
            device = negative.device
            if num_bins == 1:
                selected_neg = torch.randperm(negative.numel(), device=device)[:num_neg]
            else:
                num_per_bin = num_neg // num_bins
                negative_gt_ious = proposals_per_image.gt_ious[negative]
                min_iou, max_iou = 0, negative_gt_ious.max()
                iou_thresholds = [i * (max_iou - min_iou) / num_bins + min_iou for i in range(num_bins + 1)]
                selected_neg = []
                for i in range(num_bins):
                    filtered = nonzero_tuple((negative_gt_ious >= iou_thresholds[i]) &
                                             (negative_gt_ious < iou_thresholds[i + 1]))[0]
                    if filtered.numel() > num_per_bin:
                        filtered = filtered[torch.randperm(filtered.numel(), device=device)[:num_per_bin]]
                    selected_neg += filtered.tolist()

                if len(selected_neg) < num_neg:
                    num_extra = num_neg - len(selected_neg)
                    remains = torch.tensor(list(set(range(negative.numel())) - set(selected_neg)), device=device)
                    remains_filtered = remains[torch.randperm(remains.numel(), device=device)[:num_extra]]
                    selected_neg += remains_filtered.tolist()
            negative = negative[selected_neg]

        filtered_proposals.append(
            Instances.cat([proposals_per_image[positive], proposals_per_image[negative]])
        )
    return filtered_proposals


@ROI_HEADS_REGISTRY.register()
class ExtendedROIHeads(StandardROIHeads):
    """
    ExtendedROIHeads RoI Heads
    """

    @configurable
    def __init__(
        self,
        *,
        embed_in_features: Optional[List[str]] = None,
        embed_pooler: Optional[ROIPooler] = None,
        embed_head: Optional[nn.Module] = None,
        embed_batch_size_per_image: Optional[int] = 256,
        embed_positive_ratio: Optional[float] = 0.5,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            embed_in_features (list[str]): list of feature names to use for the embed
                pooler or embed head. None if not using embed head.
            embed_pooler (ROIPooler): pooler to extract region features from image features.
                The embed head will then take region features to make predictions.
                If None, the embed head will directly take the dict of image features
                defined by `embed_in_features`
            embed_head (nn.Module): transform features to make embed predictions
            embed_batch_size_per_image (int): number of proposals to sample for embed training
            embed_positive_fraction (float): fraction of positive (foreground) proposals
                to sample for embed training.
        """
        super().__init__(**kwargs)
        self.embed_on = embed_in_features is not None
        if self.embed_on:
            self.embed_in_features = embed_in_features
            self.embed_pooler = embed_pooler
            self.embed_head = embed_head
            self.embed_batch_size_per_image = embed_batch_size_per_image
            self.embed_positive_ratio = embed_positive_ratio

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_embed_head):
            ret.update(cls._init_embed_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_embed_head(cls, cfg, input_shape):
        if not cfg.MODEL.EMBED_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_EMBED_HEAD.IN_FEATURES
        if not in_features:
            in_features   = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_EMBED_HEAD.POOLER_RESOLUTION
        pooler_scales     = list(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_EMBED_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_EMBED_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = dict(embed_in_features=in_features)
        ret["embed_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["embed_head"] = build_embed_head(cfg, shape)
        ret["embed_batch_size_per_image"] = cfg.MODEL.ROI_EMBED_HEAD.BATCH_SIZE_PER_IMAGE
        ret["embed_positive_ratio"] = cfg.MODEL.ROI_EMBED_HEAD.POSITIVE_FRACTION
        return ret

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        See :class:`ROIHeads.label_and_sample_proposals`.

        On the basis of `ROIHeads.label_and_sample_proposals`, add `gt_ids`
        attribute to the returned results for instance association.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if self.embed_on:
                # Store matched iou for embed proposals sampling.
                if match_quality_matrix.numel() == 0:
                    matched_ious = match_quality_matrix.new_full(
                        (match_quality_matrix.size(1),), 0, dtype=torch.int64
                    )
                else:
                    matched_ious, _ = match_quality_matrix.max(dim=0)
                gt_ious = matched_ious[sampled_idxs]

                # Update gt_ids in a similar way with gt_classes.
                gt_ids = targets_per_image.gt_ids
                if gt_ids.numel() > 0:
                    gt_ids = gt_ids[matched_idxs]
                    gt_ids[matched_labels == 0] = -1
                    gt_ids[matched_labels == -1] = -1
                else:
                    gt_ids = torch.full_like(matched_idxs, -1)
                gt_ids = gt_ids[sampled_idxs]

                proposals_per_image.gt_ids = gt_ids
                proposals_per_image.gt_ious = gt_ious

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def reset_tracking(self):
        if self.embed_on:
            self.embed_head.reset_tracking()

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        losses = {}
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses.update(self._forward_box(features, proposals))
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_embed(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask, embed and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        See :class:`StandardROIHeads.forward_with_given_boxes`.

        Add `_forward_embed` function on the basis of `StandardROIHeads.forward_with_given_boxes`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_embed(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_embed(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the embedding prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict embeddings.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_embeds" and return it.
        """
        if not self.embed_on:
            # https://github.com/pytorch/pytorch/issues/49728
            if self.training:
                return {}
            else:
                return instances

        if self.training:
            # follow the implementation of qdtrack to resample proposals
            # note instances have been filtered before heads
            instances = sample_proposals_for_embed_head(
                instances, self.embed_batch_size_per_image, self.embed_positive_ratio)

        if self.embed_pooler is not None:
            features = [features[f] for f in self.embed_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.embed_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.embed_in_features}
        return self.embed_head(features, instances)
