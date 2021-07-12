import os

from detectron2.data import MetadataCatalog

from .video_coco import register_video_coco_instances


# ==== Predefined datasets and splits for video COCO ==========

_PREDEFINED_SPLITS_BDD100K_MOT = {
    "bdd100k_mot_train": (
        "bdd100k/images/track/train", "bdd100k/bdd100k_box_track_20_train.json", "bdd100k/labels/box_track_20/train"),
    "bdd100k_mot_val": (
        "bdd100k/images/track/val", "bdd100k/bdd100k_box_track_20_val.json", "bdd100k/labels/box_track_20/val"),
    "bdd100k_mot_val_part": (
        "bdd100k/images/track/val", "bdd100k/bdd100k_box_track_20_val_videos4.json", "bdd100k/labels/box_track_20/val"),
    "bdd100k_det_mot_train": (
        "bdd100k/images/100k/train", "bdd100k/bdd100k_det_mot_train.json", None),
}

metadata_bdd100k_mot = {
    "thing_classes": ["pedestrian", "rider", "car", "bus", "truck", "train", "motorcycle", "bicycle"],
    "evaluator_type": "bdd100k_mot",
}


def register_all_video_coco(root="datasets"):
    for key, (image_root, json_file, gt_folder) in _PREDEFINED_SPLITS_BDD100K_MOT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_video_coco_instances(
            key,
            metadata_bdd100k_mot,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            list(range(-4, 5))
        )
        MetadataCatalog.get(key).set(
            gt_folder=os.path.join(root, gt_folder) if gt_folder else None,
        )


register_all_video_coco()
