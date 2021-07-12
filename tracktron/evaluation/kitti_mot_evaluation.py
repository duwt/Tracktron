import copy
import os
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


class KittiMOTEvaluator(DatasetEvaluator):
    """
    Evaluate Kitti Multi Object Tracking (MOT)
    It contains a synchronization, therefore has to be called from all ranks.

    Note that the concept of AP can be implemented in different ways and may not
    produce identical results. This class mimics the implementation of the official
    Pascal VOC Matlab API, and should produce similar but not identical results to the
    official API.
    """

    def __init__(self, dataset_name, output_dir, inference_only=False):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "kitti_mot_val"
        """
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._inference_only = inference_only
        self._metadata = MetadataCatalog.get(dataset_name)
        self._seqmap = self._metadata.seqmap
        self._gt_folder = self._metadata.gt_folder

        self._logger = logging.getLogger(__name__)
        self._cpu_device = torch.device("cpu")

        self._predictions = defaultdict(list)  # video_name -> list of predictions
        self._reverse_id_mapping = {v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()} \
            if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id") else None
        self._class_name_mapping = {
            1: "Cyclist", 2: "Pedestrian", 3: "Person", 4: "Car",
            5: "Tram", 6: "Truck", 7: "Van", 8: "Misc",
        }

    def reset(self):
        self._predictions = defaultdict(list)

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            instances = output["instances"].to(self._cpu_device)
            instances = instances[instances.pred_ids >= 0]
            if len(instances) == 0:
                continue
            video_name, frame_id = input["video_name"], input["frame_id"]
            pred_boxes = instances.pred_boxes.tensor.tolist()
            pred_scores = instances.scores.tolist()
            pred_classes = instances.pred_classes.tolist()
            if self._reverse_id_mapping:
                pred_classes = [self._reverse_id_mapping[pred_class] for pred_class in pred_classes]
            pred_ids = instances.pred_ids.tolist()
            results = [
                [frame_id, pred_id, self._class_name_mapping[pred_class], *pred_box, pred_score]
                for pred_id, pred_class, pred_box, pred_score in zip(pred_ids, pred_classes, pred_boxes, pred_scores)
            ]
            self._predictions[video_name] += results

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
            video_predictions.sort()
            results_txt = ["{0} {1} {2} -1 -1 -1 {3} {4} {5} {6} -1 -1 -1 -1 -1 -1 -1 {7}\n".format(*result)
                           for result in video_predictions]
            with open(os.path.join(self._output_dir, video_name + ".txt"), "w") as f:
                f.writelines(results_txt)

        if self._inference_only:
            logger.info("All inference results have been saved in %s" % self._output_dir)
            return {}

        # TODO save evaluation results as dict
        self._results = OrderedDict()
        self._results = kitti_mot_eval(
            GT_FOLDER=self._gt_folder, TRACKERS_FOLDER=self._output_dir, SEQMAP_FILE=self._seqmap,
            TRACKERS_TO_EVAL=[""], TRACKER_SUB_FOLDER="", TIME_PROGRESS=False, PRINT_CONFIG=False
        )

        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


def kitti_mot_eval(**kwargs):
    freeze_support()

    # get default configs and override
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    config.update(kwargs)

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run evaluation
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.Kitti2DBox(dataset_config)]
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
        "Identity": ["IDF1", "IDP", "IDR"],
        "Count": ["Dets", "GT_Dets", "IDs", "GT_IDs"],
    }
    for category in ["car", "pedestrian"]:
        filtered_res[category] = OrderedDict()
        res = output_res["Kitti2DBox"][""]["COMBINED_SEQ"][category]
        for metric, sub_metrics in metrics.items():
            for sub_metric in sub_metrics:
                value = res[metric][sub_metric]
                if not isinstance(value, (int, float)):
                    value = value.mean()
                if metric != "Count":
                    value *= 100
                filtered_res[category][sub_metric] = value
    return filtered_res
