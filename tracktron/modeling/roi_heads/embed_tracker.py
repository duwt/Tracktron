from typing import List
import torch
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.structures import Instances, pairwise_iou
from detectron2.utils.registry import Registry

from tracktron.structures import TrackletStates

TRACKER_REGISTRY = Registry("TRACKER")
TRACKER_REGISTRY.__doc__ = """
Registry for tracker in a Tracker R-CNN model.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`Tracker`.
"""


def build_tracker(cfg):
    """
    Build Tracker defined by `cfg.MODEL.TRACKER.NAME`.
    """
    name = cfg.MODEL.TRACKER.NAME
    return TRACKER_REGISTRY.get(name)(cfg)


class Tracker:
    """
    A template tracker.
    """

    def reset(self):
        """
        Reset tracklets at the first video frame.
        """
        raise NotImplementedError

    def __call__(self, instances: List[Instances]):
        return self.track(instances)

    def track(self, instances: List[Instances]):
        """
        Implement tracking logic.

        Args:
            instances: instances predicted by track network.

        Returns:
            Instances: filtered instances with an extra field "pred_ids"
        """
        raise NotImplementedError


@TRACKER_REGISTRY.register()
class QuasiDenseTracker(Tracker):
    """
    Modified from https://github.com/SysCV/qdtrack/blob/master/qdtrack/models/trackers/quasi_dense_embed_tracker.py
    """

    @configurable
    def __init__(
        self,
        init_score_thr=0.7,
        obj_score_thr=0.3,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric="bi-softmax"
    ):
        """
        TODO

        Args:
            init_score_thr:
            obj_score_thr:
            match_score_thr:
            memo_tracklet_frames:
            memo_backdrop_frames:
            memo_momentum:
            nms_conf_thr:
            nms_backdrop_iou_thr:
            nms_class_iou_thr:
            with_cats:
            match_metric:
        """
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_cats = with_cats
        assert match_metric in ["bi-softmax", "softmax", "cosine"]
        self.match_metric = match_metric

        self.frame_id = 0
        self.num_tracklets = 0
        self.tracklets = dict()
        self.backdrops = []

    @classmethod
    def from_config(cls, cfg):
        return {
            "init_score_thr": cfg.MODEL.TRACKER.INIT_SCORE_THRESH,
            "obj_score_thr": cfg.MODEL.TRACKER.OBJ_SCORE_THRESH,
            "match_score_thr": cfg.MODEL.TRACKER.MATCH_SCORE_THRESH,
            "memo_tracklet_frames": cfg.MODEL.TRACKER.MEMO_TRACKLET_FRAMES,
            "memo_backdrop_frames": cfg.MODEL.TRACKER.MEMO_BACKDROP_FRAMES,
            "memo_momentum": cfg.MODEL.TRACKER.MEMO_MOMENTUM,
            "nms_conf_thr": cfg.MODEL.TRACKER.NMS_CONF_THRESH,
            "nms_backdrop_iou_thr": cfg.MODEL.TRACKER.NMS_BACKDROP_IOU_THRESH,
            "nms_class_iou_thr": cfg.MODEL.TRACKER.NMS_CLASS_IOU_THRESH,
            "with_cats": cfg.MODEL.TRACKER.WITH_CATS,
            "match_metric": cfg.MODEL.TRACKER.MATCH_METRIC,
        }

    @property
    def empty(self):
        return False if self.tracklets else True

    def reset(self):
        self.frame_id = 0
        self.num_tracklets = 0
        self.tracklets = dict()
        self.backdrops = []

    def update_memo(self, instances: Instances):
        # update memo
        filtered_instances = instances[instances.pred_ids > -1]
        ids = filtered_instances.pred_ids
        boxes = filtered_instances.pred_boxes.tensor
        embeds = filtered_instances.pred_embeds
        labels = filtered_instances.pred_classes
        for id, box, embed, label in zip(ids, boxes, embeds, labels):
            id = int(id)
            if id in self.tracklets.keys():
                acc_frame = self.tracklets[id]["acc_frame"] + 1
                embed = (1 - self.memo_momentum) * self.tracklets[id]["embed"] + self.memo_momentum * embed
                velocity = (box - self.tracklets[id]["box"]) / (self.frame_id - self.tracklets[id]["last_frame"])
                velocity = (self.tracklets[id]["velocity"] * (acc_frame - 1) + velocity) / acc_frame
            else:
                acc_frame = 0
                velocity = torch.zeros_like(box)
            self.tracklets[id] = dict(
                box=box, embed=embed, label=label, last_frame=self.frame_id,
                velocity=velocity, acc_frame=acc_frame
            )

        backdrop_indices = torch.nonzero(instances.pred_ids == -1, as_tuple=False).squeeze(1)
        overlaps = pairwise_iou(instances.pred_boxes[backdrop_indices], instances.pred_boxes)
        for i, ind in enumerate(backdrop_indices):
            if (overlaps[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_indices[i] = -1
        backdrop_indices = backdrop_indices[backdrop_indices > -1]
        backdrop_instances = instances[backdrop_indices]

        self.backdrops.insert(
            0, dict(
                boxes=backdrop_instances.pred_boxes.tensor,
                embeds=backdrop_instances.pred_embeds,
                labels=backdrop_instances.pred_classes
            )
        )

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if self.frame_id - v["last_frame"] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

        # update frame_id
        self.frame_id += 1

    @property
    def memo(self):
        memo_embeds = []
        memo_ids = []
        memo_boxes = []
        memo_labels = []
        memo_vs = []
        for tracklet_id, tracklet in self.tracklets.items():
            memo_boxes.append(tracklet["box"][None, :])
            memo_embeds.append(tracklet["embed"][None, :])
            memo_ids.append(tracklet_id)
            memo_labels.append(tracklet["label"].view(1, 1))
            memo_vs.append(tracklet["velocity"][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        for backdrop in self.backdrops:
            backdrop_ids = torch.full((1, backdrop["embeds"].size(0)), -1, dtype=torch.long)
            backdrop_vs = torch.zeros_like(backdrop["boxes"])
            memo_boxes.append(backdrop["boxes"])
            memo_embeds.append(backdrop["embeds"])
            memo_ids = torch.cat([memo_ids, backdrop_ids], dim=1)
            memo_labels.append(backdrop["labels"][:, None])
            memo_vs.append(backdrop_vs)

        memo_boxes = torch.cat(memo_boxes, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_vs = torch.cat(memo_vs, dim=0)
        return memo_boxes, memo_labels, memo_embeds, memo_ids.squeeze(0), memo_vs

    def track(self, instances):
        results = []
        for instances_per_frame in instances:
            scores, indices = instances_per_frame.scores.sort(descending=True)
            instances_per_frame = instances_per_frame[indices]
            boxes = instances_per_frame.pred_boxes

            # duplicate removal for potential backdrops and cross classes
            valid = boxes.tensor.new_ones(len(boxes))
            overlaps = pairwise_iou(boxes, boxes)
            for i in range(1, len(boxes)):
                thr = self.nms_backdrop_iou_thr if scores[i] < self.obj_score_thr else self.nms_class_iou_thr
                if (overlaps[i, :i] > thr).any():
                    valid[i] = 0
            valid = valid == 1

            scores = instances_per_frame.scores[valid]
            boxes = instances_per_frame.pred_boxes[valid]
            labels = instances_per_frame.pred_classes[valid]
            embeds = instances_per_frame.pred_embeds[valid]

            # init ids container
            ids = torch.full_like(labels, -1)

            # match if buffer is not empty
            if len(boxes) > 0 and not self.empty:
                _, memo_labels, memo_embeds, memo_ids, _ = self.memo

                if self.match_metric == "bi-softmax":
                    similarity = torch.mm(embeds, memo_embeds.t())
                    d2t_match_scores = similarity.softmax(dim=1)
                    t2d_match_scores = similarity.softmax(dim=0)
                    match_scores = (d2t_match_scores + t2d_match_scores) / 2
                elif self.match_metric == "softmax":
                    similarity = torch.mm(embeds, memo_embeds.t())
                    match_scores = similarity.softmax(dim=1)
                elif self.match_metric == "cosine":
                    match_scores = torch.mm(
                        F.normalize(embeds, p=2, dim=1),
                        F.normalize(memo_embeds, p=2, dim=1).t()
                    )
                else:
                    raise NotImplementedError

                if self.with_cats:
                    cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                    match_scores *= cat_same.float()

                for i in range(len(boxes)):
                    match_score, memo_ind = torch.max(match_scores[i, :], dim=0)
                    id = memo_ids[memo_ind]
                    if match_score > self.match_score_thr:
                        if id > -1:
                            if scores[i] > self.obj_score_thr:
                                ids[i] = id
                                match_scores[:i, memo_ind] = 0
                                match_scores[i + 1:, memo_ind] = 0
                            elif match_score > self.nms_conf_thr:
                                ids[i] = -2
            new_indices = (ids == -1) & (scores > self.init_score_thr)
            num_news = new_indices.sum()
            ids[new_indices] = torch.arange(
                self.num_tracklets,
                self.num_tracklets + num_news,
                dtype=torch.long, device=ids.device
            )
            self.num_tracklets += num_news

            instances_per_frame.pred_ids = torch.full_like(instances_per_frame.pred_classes, -1)
            instances_per_frame.pred_ids[valid] = ids
            self.update_memo(instances_per_frame[valid])
            results.append(instances_per_frame)
        return results
