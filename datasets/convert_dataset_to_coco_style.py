import os
import json
import argparse
from functools import partial


# COCO annotation format (video_id, frame_id, instance_id for video tasks):
# frame_id should be a sequence [0, N] or having the first_frame field
# dict(
#     info: dict(description, url, version, year, contributor, date_created),
#     licenses: [dict(url, id, name)],
#     images: [dict(license, file_name, coco_url, height, width, video_name,
#                   video_id, frame_id, date_captured, flickr_url, id)],
#     annotations: [dict(segmentation, area, iscrowd, image_id, bbox,
#                        category_id, id, instance_id)],
#     categories: [dict(supercategory, id, name)]
# )


def convert_bdd100k_mot_to_coco_style(anno_dir, out_file):
    assert anno_dir and out_file
    categories = [
        {"id": 1, "name": "pedestrian", "supercategory": "human"},
        {"id": 2, "name": "rider", "supercategory": "human"},
        {"id": 3, "name": "car", "supercategory": "vehicle"},
        {"id": 4, "name": "bus", "supercategory": "vehicle"},
        {"id": 5, "name": "truck", "supercategory": "vehicle"},
        {"id": 6, "name": "train", "supercategory": "vehicle"},
        {"id": 7, "name": "motorcycle", "supercategory": "bike"},
        {"id": 8, "name": "bicycle", "supercategory": "bike"},
    ]
    category_mapping = {
        "pedestrian": 1, "rider": 2, "car": 3, "bus": 4,
        "truck": 5, "train": 6, "motorcycle": 7, "bicycle": 8
    }
    ignore_categories = {"other person", "trailer", "other vehicle"}
    images, annotations = [], []
    for video_id, anno_file in enumerate(sorted(os.listdir(anno_dir))):
        sequence_annos = json.load(open(os.path.join(anno_dir, anno_file)))
        for annos in sequence_annos:
            video_name = annos["videoName"]
            frame_id = annos["frameIndex"]
            image_id = len(images) + 1
            file_name = os.path.join(video_name, annos["name"])
            width, height = 1280, 720
            images.append(
                dict(file_name=file_name, video_id=video_id, video_name=video_name,
                     frame_id=frame_id, height=height, width=width, id=image_id)
            )
            for anno in annos["labels"]:
                instance_id = anno["id"]
                category = anno["category"]
                if category in ignore_categories:
                    continue
                class_id = category_mapping[category]
                box = anno["box2d"]
                bbox = [box["x1"], box["y1"], box["x2"] - box["x1"], box["y2"] - box["y1"]]
                annotations.append(
                    dict(area=bbox[2] * bbox[3], id=len(annotations) + 1,
                         image_id=image_id, bbox=bbox, category_id=class_id,
                         instance_id=instance_id, iscrowd=anno["attributes"]["crowd"])
                )
    dst_annos = dict(images=images, annotations=annotations, categories=categories)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(dst_annos, f)


def convert_bdd100k_det_to_coco_style(anno_file, out_file):
    assert anno_file and out_file
    categories = [
        {"id": 1, "name": "pedestrian", "supercategory": "human"},
        {"id": 2, "name": "rider", "supercategory": "human"},
        {"id": 3, "name": "car", "supercategory": "vehicle"},
        {"id": 4, "name": "bus", "supercategory": "vehicle"},
        {"id": 5, "name": "truck", "supercategory": "vehicle"},
        {"id": 6, "name": "train", "supercategory": "vehicle"},
        {"id": 7, "name": "motorcycle", "supercategory": "bike"},
        {"id": 8, "name": "bicycle", "supercategory": "bike"},
    ]
    category_mapping = {
        "pedestrian": 1, "rider": 2, "car": 3, "bus": 4,
        "truck": 5, "train": 6, "motorcycle": 7, "bicycle": 8
    }
    ignore_categories = {"traffic light", "traffic sign", "other person", "trailer", "other vehicle"}
    images, annotations = [], []
    sequence_annos = json.load(open(anno_file))
    for annos in sequence_annos:
        image_id = len(images) + 1
        file_name = annos["name"]
        width, height = 1280, 720
        images.append(
            dict(file_name=file_name, video_id=-1, video_name=None,
                 frame_id=-1, height=height, width=width, id=image_id)
        )
        for anno in annos.get("labels", list()):
            category = anno["category"]
            if category in ignore_categories:
                continue
            class_id = category_mapping[category]
            box = anno["box2d"]
            bbox = [box["x1"], box["y1"], box["x2"] - box["x1"], box["y2"] - box["y1"]]
            annotations.append(
                dict(area=bbox[2] * bbox[3], id=len(annotations)+1,
                     image_id=image_id, bbox=bbox, category_id=class_id,
                     instance_id=len(annotations)+1, iscrowd=False)
            )
    dst_annos = dict(images=images, annotations=annotations, categories=categories)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(dst_annos, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert Annotations to COCO Style")
    parser.add_argument("dataset", choices=["bdd100k-det", "bdd100k-mot"], help="dataset name")
    parser.add_argument("--image_dir", type=str, help="base directory of annotations")
    parser.add_argument("--anno_dir", type=str, help="base directory of annotations")
    parser.add_argument("--anno_file", type=str, help="annotation file for dataset")
    parser.add_argument("--base_dir", type=str, help="base directory for specific dataset like mot-challenge")
    parser.add_argument("--out_file", type=str, help="output annotation file path")
    parser.add_argument("--out_dir", type=str, help="directory of output annotation files")
    parser.add_argument("--mode", type=str, default="full", choices=["half0", "half1", "full"],
                        help="dataset split for specific dataset")
    args = parser.parse_args()

    convert_func_dict = {
        "bdd100k-mot": partial(convert_bdd100k_mot_to_coco_style, args.anno_dir, args.out_file),
        "bdd100k-det": partial(convert_bdd100k_det_to_coco_style, args.anno_file, args.out_file),
    }

    convert_func_dict[args.dataset]()
