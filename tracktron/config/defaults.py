from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# Options: "Trainer", "VideoTrainer"
_C.TRAINER = "VideoTrainer"
# If True, save inference output but not evaluate
_C.INFERENCE_ONLY = False
_C.MODEL.EMBED_ON = False
_C.INPUT.NUM_FRAMES_TRAIN = 2

# ---------------------------------------------------------------------------- #
# DataLoader for Video Trainer
# ---------------------------------------------------------------------------- #
# Use "VideoTrainingSampler" for video trainer
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
_C.DATALOADER.SAMPLER_PAIR_OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
_C.DATALOADER.SAMPLER_CUR_FRAME_WEIGHT = 0


# ---------------------------------------------------------------------------- #
# RoI Embed Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_EMBED_HEAD = CN()
_C.MODEL.ROI_EMBED_HEAD.NAME = "QuasiDenseEmbedHead"
# Names of the input feature maps to be used by embed head
# if empty, use the default features of RoI heads
_C.MODEL.ROI_EMBED_HEAD.IN_FEATURES = []
_C.MODEL.ROI_EMBED_HEAD.NUM_FC = 1
_C.MODEL.ROI_EMBED_HEAD.FC_DIM = 1024
_C.MODEL.ROI_EMBED_HEAD.NUM_CONV = 4
_C.MODEL.ROI_EMBED_HEAD.CONV_DIM = 256
_C.MODEL.ROI_EMBED_HEAD.NORM = "GN"
_C.MODEL.ROI_EMBED_HEAD.EMBED_CHANNELS = 256

_C.MODEL.ROI_EMBED_HEAD.EMBED_LOSS_WEIGHT = 0.25
_C.MODEL.ROI_EMBED_HEAD.EMBED_AUX_LOSS = True
_C.MODEL.ROI_EMBED_HEAD.EMBED_AUX_LOSS_WEIGHT = 1
_C.MODEL.ROI_EMBED_HEAD.BIDIRECTIONAL_LOSS = False
_C.MODEL.ROI_EMBED_HEAD.INTRA_FRAME_LOSS = False
_C.MODEL.ROI_EMBED_HEAD.CLS_AGNOSTIC_EMBED = True
_C.MODEL.ROI_EMBED_HEAD.CLS_AGNOSTIC_EMBED_LOSS = True

_C.MODEL.ROI_EMBED_HEAD.POOLER_RESOLUTION = 7
_C.MODEL.ROI_EMBED_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_EMBED_HEAD.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.ROI_EMBED_HEAD.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.ROI_EMBED_HEAD.POSITIVE_FRACTION = 0.5


# ---------------------------------------------------------------------------- #
# Embed Tracker
# ---------------------------------------------------------------------------- #
_C.MODEL.TRACKER = CN()
_C.MODEL.TRACKER.NAME = "QuasiDenseTracker"
_C.MODEL.TRACKER.INIT_SCORE_THRESH = 0.7
_C.MODEL.TRACKER.OBJ_SCORE_THRESH = 0.3
_C.MODEL.TRACKER.MATCH_SCORE_THRESH = 0.5
_C.MODEL.TRACKER.MEMO_TRACKLET_FRAMES = 10
_C.MODEL.TRACKER.MEMO_BACKDROP_FRAMES = 1
_C.MODEL.TRACKER.MEMO_MOMENTUM = 0.8
_C.MODEL.TRACKER.NMS_CONF_THRESH = 0.5
_C.MODEL.TRACKER.NMS_BACKDROP_IOU_THRESH = 0.3
_C.MODEL.TRACKER.NMS_CLASS_IOU_THRESH = 0.7
_C.MODEL.TRACKER.WITH_CATS = True
_C.MODEL.TRACKER.MATCH_METRIC = "bi-softmax"
