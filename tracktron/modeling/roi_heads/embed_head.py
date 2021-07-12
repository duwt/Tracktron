from typing import List, Union, Dict
import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm, cat
from detectron2.structures import Instances
from detectron2.utils.registry import Registry

from .embed_tracker import build_tracker, Tracker

ROI_EMBED_HEAD_REGISTRY = Registry("ROI_EMBED_HEAD")
ROI_EMBED_HEAD_REGISTRY.__doc__ = """
Registry for embed heads, which predicts instance embeddings given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


class BaseEmbedHead(nn.Module):
    """
    Implement the basic embed losses and inference logic.
    """

    @configurable
    def __init__(self, *, tracker=None, vis_period=0):
        """
        NOTE: this interface is experimental.

        Args:
            tracker (Tracker): a tracker that predicts instance ids based on embeddings
            vis_period (int): visualization period
        """
        super().__init__()
        self.tracker = tracker
        self.vis_period = vis_period

    @classmethod
    def from_config(cls, cfg):
        return {
            "tracker": build_tracker(cfg),
            "vis_period": cfg.VIS_PERIOD,
        }

    def reset_tracking(self):
        self.tracker.reset()

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" with an extra
                "pred_embeds" field storing embedding vectors in inference.
        """
        x = self.layers(x)

        cls_agnostic_embed = x.size(1) == 1
        if cls_agnostic_embed:
            x = x.squeeze(1)
        else:
            if self.training:
                classes = cat([instances_i.gt_classes for instances_i in instances])
            else:
                classes = cat([instances_i.pred_classes for instances_i in instances])
            indices = torch.arange(x.size(0), device=classes.device)
            x = x[indices, classes]

        if self.training:
            return self.losses(x, instances)
        else:
            embeds = x.split([len(instances_i) for instances_i in instances])
            for embeds_i, instances_i in zip(embeds, instances):
                instances_i.pred_embeds = embeds_i
            instances = self.tracker(instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError

    def losses(self, x, instances: List[Instances]):
        """
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        raise NotImplementedError


@ROI_EMBED_HEAD_REGISTRY.register()
class QuasiDenseEmbedHead(BaseEmbedHead, nn.Sequential):
    """
    An implementation based on https://arxiv.org/abs/2006.06664
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        conv_dims: List[int],
        fc_dims: List[int],
        conv_norm: str,
        num_classes: int,
        embed_channels: int = 256,
        embed_aux_loss: bool = True,
        bidirectional_loss: bool = False,
        intra_frame_loss: bool = False,
        class_agnostic_embed: bool = True,
        class_agnostic_loss: bool = True,
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
            num_classes (int): the number of foreground classes (i.e. background
                is not included).
            embed_channels (int): the output embedding dimensions
            embed_aux_loss (bool): whether to apply auxiliary l2 loss
            bidirectional_loss (bool): whether to apply bidirectional embed loss
            intra_frame_loss (bool): whether to apply intra frame loss
            class_agnostic_embed (bool): whether to predict agnostic embed
            class_agnostic_loss (bool): whether to apply class agnostic loss
            loss_weight (float or dict): weights to use for losses. Can be single
                float for weighting all losses, or a dict of individual weightings.
                Valid dict keys are:
                    * "loss_embed": applied to multi-positive softmax loss
                    * "loss_embed_aux": applied to auxiliary l2 loss
        """
        super().__init__(**kwargs)
        assert len(conv_dims) + len(fc_dims) > 0

        self.num_classes = num_classes
        self.embed_channels = embed_channels
        self.class_agnostic_embed = class_agnostic_embed
        self.class_agnostic_loss = class_agnostic_loss
        self.num_embeds = 1 if self.class_agnostic_embed else self.num_classes + 1
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(inplace=True),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU(inplace=True))
            self.fcs.append(fc)
            self._output_size = fc_dim

        self.predictor = Linear(int(np.prod(self._output_size)), embed_channels * self.num_embeds)

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)
        nn.init.normal_(self.predictor.weight, 0, 0.01)
        nn.init.constant_(self.predictor.bias, 0)

        self.embed_aux_loss = embed_aux_loss
        if isinstance(loss_weight, float):
            loss_weight = {"loss_embed": loss_weight, "loss_embed_aux": loss_weight}
        self.loss_weight = loss_weight
        self.bidirectional_loss = bidirectional_loss and (not intra_frame_loss)
        self.intra_frame_loss = intra_frame_loss

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        assert cfg.INPUT.NUM_FRAMES_TRAIN == 2, "Only two frames training is supported."
        num_conv = cfg.MODEL.ROI_EMBED_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_EMBED_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_EMBED_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_EMBED_HEAD.FC_DIM
        ret.update({
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_EMBED_HEAD.NORM,
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "embed_channels": cfg.MODEL.ROI_EMBED_HEAD.EMBED_CHANNELS,
            "embed_aux_loss": cfg.MODEL.ROI_EMBED_HEAD.EMBED_AUX_LOSS,
            "bidirectional_loss": cfg.MODEL.ROI_EMBED_HEAD.BIDIRECTIONAL_LOSS,  # TODO
            "intra_frame_loss": cfg.MODEL.ROI_EMBED_HEAD.INTRA_FRAME_LOSS,
            "class_agnostic_embed": cfg.MODEL.ROI_EMBED_HEAD.CLS_AGNOSTIC_EMBED,
            "class_agnostic_loss": cfg.MODEL.ROI_EMBED_HEAD.CLS_AGNOSTIC_EMBED_LOSS,  # TODO
            "loss_weight": {
                "loss_embed": cfg.MODEL.ROI_EMBED_HEAD.EMBED_LOSS_WEIGHT,
                "loss_embed_aux": cfg.MODEL.ROI_EMBED_HEAD.EMBED_AUX_LOSS_WEIGHT
            },
        })
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x.view(x.size(0), self.num_embeds, self.embed_channels)

    def losses(self, x, instances: List[Instances]):
        # split features
        embeds = x.split([len(instances_i) for instances_i in instances])
        N = len(instances) // 2
        key_embeds, ref_embeds = embeds[:N], embeds[N:]
        key_gt_ids = [inst.gt_ids for inst in instances[:N]]
        ref_gt_ids = [inst.gt_ids for inst in instances[N:]]
        key_gt_classes = [inst.gt_classes for inst in instances[:N]]
        ref_gt_classes = [inst.gt_classes for inst in instances[N:]]

        # filter positive instances for key frame
        key_embeds_pos = [embeds_i[gt_ids_i >= 0] for embeds_i, gt_ids_i in zip(key_embeds, key_gt_ids)]
        key_gt_classes_pos = [gt_classes_i[gt_ids_i >= 0] for gt_ids_i, gt_classes_i in zip(key_gt_ids, key_gt_classes)]
        key_gt_ids_pos = [gt_ids_i[gt_ids_i >= 0] for gt_ids_i in key_gt_ids]

        # set instances in key frame as part of reference
        if self.intra_frame_loss:
            ref_embeds = [torch.cat([key, ref], dim=0) for key, ref in zip(key_embeds_pos, ref_embeds)]
            ref_gt_classes = [torch.cat([key, ref]) for key, ref in zip(key_gt_classes_pos, ref_gt_classes)]
            ref_gt_ids = [torch.cat([key, ref]) for key, ref in zip(key_gt_ids_pos, ref_gt_ids)]

        if sum([min(key_embed.size(0), ref_embed.size(0))
                for key_embed, ref_embed in zip(key_embeds_pos, ref_embeds)]) == 0:
            losses = dict()
            losses["loss_embed"] = x.sum() * 0
            if self.embed_aux_loss:
                losses["loss_embed_aux"] = x.sum() * 0
            return losses

        # dot product similarity and cosine similarity
        dists = self.get_similarity(key_embeds_pos, ref_embeds, normalize=False)
        cos_dists = self.get_similarity(key_embeds_pos, ref_embeds, normalize=True)

        pos_targets, neg_targets = [], []
        for key_gt_id, ref_gt_id, ket_gt_class, ref_gt_class in \
                zip(key_gt_ids_pos, ref_gt_ids, key_gt_classes_pos, ref_gt_classes):
            inst_id_match = key_gt_id.view(-1, 1) == ref_gt_id.view(1, -1)
            cls_id_match = ket_gt_class.view(-1, 1) == ref_gt_class.view(1, -1)
            pos_target = (inst_id_match & cls_id_match).float()
            neg_target = 1 - pos_target if self.class_agnostic_loss else ((~inst_id_match) & cls_id_match).float()
            pos_targets.append(pos_target)
            neg_targets.append(neg_target)

        losses = dict()
        loss_embed = self.embed_loss_func(dists, pos_targets, neg_targets)
        losses["loss_embed"] = loss_embed
        if self.embed_aux_loss:
            loss_embed_aux = self.embed_aux_loss_func(cos_dists, pos_targets, neg_targets)
            losses["loss_embed_aux"] = loss_embed_aux

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    @staticmethod
    def embed_loss_func(dists, pos_targets, neg_targets):
        loss_embed = []
        for dist, pos_target, neg_target in zip(dists, pos_targets, neg_targets):
            if dist.numel() == 0:
                continue
            exp_pos = (torch.exp(-dist) * pos_target).sum(dim=1)
            exp_neg = (torch.exp(dist.clamp(max=80)) * neg_target).sum(dim=1)
            loss = torch.log(1 + exp_pos * exp_neg)
            weight = pos_target.sum(dim=1).clamp(max=1)
            loss_embed.append((loss * weight).sum() / weight.sum().clamp(min=1))
        return sum(loss_embed) / len(loss_embed)

    @staticmethod
    def embed_aux_loss_func(cos_dists, pos_targets, neg_targets,
                            neg_pos_upper_ratio=3, pos_margin=0, neg_margin=0.3):
        loss_embed_aux = []
        for cos_dist, pos_target, neg_target in zip(cos_dists, pos_targets, neg_targets):
            if cos_dist.numel() == 0:
                continue
            if pos_margin > 0:
                cos_dist[pos_target > 0] += pos_margin
            if neg_margin > 0:
                cos_dist[neg_target > 0] -= neg_margin
            cos_dist = cos_dist.clamp(min=0, max=1)

            # l2 loss
            loss = (cos_dist - pos_target) ** 2

            num_pos = int(pos_target.sum())
            num_neg = int(neg_target.sum())
            if num_pos > 0 and num_neg / num_pos > neg_pos_upper_ratio:
                # hard negative mining
                num_neg = num_pos * neg_pos_upper_ratio
                loss_pos = (loss * pos_target).sum()
                loss_neg = (loss * neg_target).flatten().topk(num_neg)[0].sum()
                loss_embed_aux.append((loss_pos + loss_neg) / (num_pos + num_neg))
            else:
                loss_embed_aux.append((loss * (pos_target + neg_target)).sum() / (num_pos + num_neg))

        return sum(loss_embed_aux) / len(loss_embed_aux)

    @staticmethod
    def get_similarity(key_embeds, ref_embeds, normalize=True):
        dists = []
        for key_embed, ref_embed in zip(key_embeds, ref_embeds):
            if key_embed.size(0) == 0 or ref_embed.size(0) == 0:
                dist = torch.zeros((key_embed.size(0), ref_embed.size(0)),
                                   device=key_embed.device, dtype=torch.float32)
            else:
                if normalize:
                    key_embed = F.normalize(key_embed, p=2, dim=1)
                    ref_embed = F.normalize(ref_embed, p=2, dim=1)
                dist = torch.mm(key_embed, ref_embed.t())
            dists.append(dist)
        return dists


def build_embed_head(cfg, input_shape):
    """
    Build a embed head defined by `cfg.MODEL.ROI_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_EMBED_HEAD.NAME
    return ROI_EMBED_HEAD_REGISTRY.get(name)(cfg, input_shape)
