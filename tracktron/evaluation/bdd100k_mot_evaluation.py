import copy
import os
import json
import logging
import torch
from multiprocessing import freeze_support
from collections import defaultdict, OrderedDict

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager

from . import trackeval

logger = logging.getLogger(__name__)


class BDD100KMOTEvaluator(DatasetEvaluator):
    """
    Evaluate BDD100K Multi Object Tracking (MOT)
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name, output_dir, inference_only=False):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "bdd100k_mot_val"
        """
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._inference_only = inference_only
        self._metadata = MetadataCatalog.get(dataset_name)
        self._gt_folder = self._metadata.gt_folder

        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")

        self._predictions = defaultdict(list)  # video_name -> list of predictions
        self._reverse_id_mapping = {v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()} \
            if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id") else None
        self._class_name_mapping = {
            1: "pedestrian", 2: "rider", 3: "car", 4: "bus",
            5: "truck", 6: "train", 7: "motorcycle", 8: "bicycle"
        }

    def reset(self):
        self._predictions = defaultdict(list)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            instances = output["instances"].to(self._cpu_device)
            instances = instances[instances.pred_ids >= 0]
            video_name = input["video_name"]
            file_name, frame_id = input["file_name"], input["frame_id"]
            if len(instances) == 0:
                results = []
            else:
                pred_boxes = instances.pred_boxes.tensor.tolist()
                pred_scores = instances.scores.tolist()
                pred_classes = instances.pred_classes.tolist()
                if self._reverse_id_mapping:
                    pred_classes = [self._reverse_id_mapping[pred_class] for pred_class in pred_classes]
                pred_ids = instances.pred_ids.tolist()
                results = [
                    dict(category=self._class_name_mapping[pred_class], id=pred_id, score=pred_score,
                         box2d=dict(x1=pred_box[0], y1=pred_box[1], x2=pred_box[2], y2=pred_box[3]))
                    for pred_id, pred_class, pred_box, pred_score in zip(pred_ids, pred_classes, pred_boxes, pred_scores)
                ]
            self._predictions[video_name].append(
                dict(videoName=video_name, name=os.path.basename(file_name), frameIndex=frame_id, labels=results)
            )

    def evaluate(self):
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return {}

        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            predictions.update(predictions_per_rank)
        del all_predictions

        PathManager.mkdirs(self._output_dir)
        for video_name, video_predictions in predictions.items():
            with open(os.path.join(self._output_dir, video_name + ".json"), "w") as f:
                json.dump(video_predictions, f)

        if self._inference_only:
            logger.info("All inference results have been saved in %s" % self._output_dir)
            return {}

        # TODO save evaluation results as dict
        self._results = bdd100k_mot_eval(
            GT_FOLDER=self._gt_folder, TRACKERS_FOLDER=self._output_dir, TRACKERS_TO_EVAL=[""],
            TRACKER_SUB_FOLDER="", TIME_PROGRESS=False, PRINT_CONFIG=False
        )

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


def bdd100k_mot_eval(**kwargs):
    freeze_support()

    # get default configs and override
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['PRINT_ONLY_COMBINED'] = True
    default_dataset_config = trackeval.datasets.BDD100K.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    config.update(kwargs)

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run evaluation
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.BDD100K(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)

    # filter some metrics
    filtered_res = OrderedDict()
    metrics = {
        "HOTA": ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA"],
        "CLEAR": ["MOTA", "MOTP", "sMOTA"],
        "Identity": ["IDF1", "IDFN", "IDFP"],
        "Count": ["Dets", "GT_Dets", "IDs", "GT_IDs"],
    }
    for category in ["cls_comb_cls_av", "cls_comb_det_av"]:
        filtered_res[category] = OrderedDict()
        res = output_res["BDD100K"][""]["COMBINED_SEQ"][category]
        for metric, sub_metrics in metrics.items():
            for sub_metric in sub_metrics:
                value = res[metric][sub_metric]
                if not isinstance(value, (int, float)):
                    value = value.mean()
                if metric != "Count" and sub_metric not in ["IDFN", "IDFP"]:
                    value *= 100
                filtered_res[category][sub_metric] = value
    return filtered_res
