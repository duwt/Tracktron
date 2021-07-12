import os
import json
import random
import pathlib
import argparse
import itertools
from functools import partial
from collections import Counter

import cv2
import pandas as pd
import numpy as np
import pycocotools.mask as mask_utils
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color


VISUALIZE_CONTINUE = 0
VISUALIZE_EXIT = 1


def xywh_to_xyxy(bbox):
    return bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]


def polys_to_mask(polygons, height, width):
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask


def visualize(vis_image, wait_time=0):
    cv2.namedWindow("visualize")
    cv2.imshow("visualize", vis_image)
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord("q"):
        cv2.destroyWindow("visualize")
        return VISUALIZE_EXIT
    elif wait_time > 0 and key == ord(" "):
        cv2.waitKey(-1)
    return VISUALIZE_CONTINUE


def kitti_mot_visualize(image_dir, anno_file, wait=1, scale=1):
    """
        anno_file format:
            time_frame(1) id(1) type(1) truncated(1) occluded(1) alpha(1)
            bbox(4) dimensions(3) location(3) rotation_y(1) *score(1)
    """
    assert image_dir and anno_file
    image_list = sorted(os.listdir(image_dir))
    annos = pd.read_csv(anno_file, header=None, sep=" ", usecols=(0, 1, 2, 6, 7, 8, 9))
    annos = np.array(annos)
    anno_index = 0
    instance_color_dict = dict()
    for i, image_file in enumerate(image_list):
        image = cv2.imread(os.path.join(image_dir, image_file), cv2.IMREAD_UNCHANGED)
        labels, boxes, colors = [], [], []
        while anno_index < len(annos) and annos[anno_index][0] == i:
            instance_id = str(annos[anno_index][2]) + str(annos[anno_index][1])
            labels.append(str(instance_id))
            boxes.append(annos[anno_index][3:].astype(np.int32))
            if instance_id not in instance_color_dict:
                instance_color_dict[instance_id] = random_color(rgb=True, maximum=1)
            colors.append(instance_color_dict[instance_id])
            anno_index += 1
        visualizer = Visualizer(image, metadata=None, scale=scale)
        vis = visualizer.overlay_instances(labels=labels, boxes=boxes, assigned_colors=colors, alpha=0.9)
        state = visualize(vis.get_image(), wait_time=wait)
        if state == VISUALIZE_EXIT:
            return


def kitti_mots_visualize(image_dir, anno_file, wait=1, scale=1):
    """
        anno_file format:
            time_frame(1) id(1) class_id(1) img_height(1) img_width(1) rle(1)
    """
    assert image_dir and anno_file
    image_list = sorted(os.listdir(image_dir))
    annos = pd.read_csv(anno_file, header=None, sep=" ",
                        names=("time_frame", "id", "class_id", "height", "width", "rle"))
    anno_index = 0
    instance_color_dict = dict()
    for i, image_file in enumerate(image_list):
        image = cv2.imread(os.path.join(image_dir, image_file), cv2.IMREAD_UNCHANGED)
        labels, masks, colors = [], [], []
        while anno_index < len(annos) and annos.iloc[anno_index]["time_frame"] == i:
            instance_id = annos.iloc[anno_index]["id"]
            # if instance_id == 10000:
            #     anno_index += 1
            #     continue
            labels.append(str(instance_id))
            rle = {"size": image.shape[:2], "counts": annos.iloc[anno_index]["rle"].encode(encoding="utf-8")}
            masks.append(mask_utils.decode(rle))
            if instance_id not in instance_color_dict:
                instance_color_dict[instance_id] = random_color(rgb=True, maximum=1)
            colors.append(instance_color_dict[instance_id])
            anno_index += 1
        visualizer = Visualizer(image, metadata=None, scale=scale)
        vis = visualizer.overlay_instances(labels=labels, masks=masks, assigned_colors=colors)
        state = visualize(vis.get_image(), wait_time=wait)
        if state == VISUALIZE_EXIT:
            return


def apollo_ins_visualize(image_dir, anno_dir, scale=1, shuffle=False):
    """
        anno_file format:
            dict(imgHeight, imgWidth, objects=[dict(polygons, label)])
    """
    assert image_dir and anno_dir
    assert shuffle is False, "shuffle is not supported for a large dataset"
    for image_file in pathlib.Path(image_dir).glob("**/*.jpg"):
        image_file = str(image_file)
        anno_file = os.path.join(anno_dir, os.path.relpath(image_file, image_dir))[:-4] + ".json"
        if not os.path.isfile(anno_file):
            continue
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        annos = json.load(open(anno_file))
        height, width = image.shape[:2]
        masks = [polys_to_mask([list(itertools.chain(*polygon)) for polygon in anno["polygons"]], height, width)
                 for anno in annos["objects"]]
        labels = [str(anno["label"]) for anno in annos["objects"]]
        visualizer = Visualizer(image, metadata=None, scale=scale)
        vis = visualizer.overlay_instances(labels=labels, masks=masks)
        state = visualize(vis.get_image())
        if state == VISUALIZE_EXIT:
            return
        # image_file = str(image_file)
        # anno_file = os.path.join(anno_dir, os.path.relpath(image_file, image_dir))[:-4] + "_instanceIds.png"
        # if not os.path.isfile(anno_file):
        #     continue
        # image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        # anno = cv2.imread(anno_file, cv2.IMREAD_UNCHANGED)
        # height_crop = image.shape[0] // 2
        # image = image[height_crop:]
        # anno = anno[height_crop:]
        # counter = Counter(anno.flatten())
        # labels, masks = [], []
        # for key in counter.keys():
        #     # if key > 1000:
        #     if 33000 <= key < 41000:
        #         labels.append(str(key // 1000))
        #         masks.append(anno == key)
        # visualizer = Visualizer(image, metadata=None, scale=scale)
        # vis = visualizer.overlay_instances(labels=labels, masks=masks)
        # state = visualize(vis.get_image())
        # if state == VISUALIZE_EXIT:
        #     return


def apollo_mots_visualize(image_dir, anno_dir, wait=1, scale=1):
    assert image_dir and anno_dir
    image_list = sorted(os.listdir(image_dir))
    instance_color_dict = dict()
    for image_file in image_list:
        image_file = os.path.join(image_dir, image_file)
        anno_file = os.path.join(anno_dir, os.path.basename(image_file))[:-4] + ".png"
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        anno = cv2.imread(anno_file, cv2.IMREAD_UNCHANGED)
        counter = Counter(anno.flatten())
        labels, masks, colors = [], [], []
        for key in counter.keys():
            if key > 1000:
                labels.append(key)
                masks.append(anno == key)
                if key not in instance_color_dict:
                    instance_color_dict[key] = random_color(rgb=True, maximum=1)
                colors.append(instance_color_dict[key])
        visualizer = Visualizer(image, metadata=None, scale=scale)
        vis = visualizer.overlay_instances(labels=labels, masks=masks, assigned_colors=colors)
        state = visualize(vis.get_image(), wait_time=wait)
        if state == VISUALIZE_EXIT:
            return


def bdd100k_mot_visualize(image_dir, anno_file, wait=1, scale=1):
    """
        anno_file format:
            [dict(
                name, videoName, frameIndex, labels=[dict(
                    id, category, box2d=dict(x1, x2, y1, y2),
                    attributes=dict(occluded, truncated, crowd)
                )]
            )]
    """
    assert image_dir and anno_file
    image_list = sorted(os.listdir(image_dir))
    annos_list = json.load(open(anno_file))
    assert len(image_list) == len(annos_list)
    instance_color_dict = dict()
    for image_file, annos in zip(image_list, annos_list):
        assert image_file == annos["name"]
        image_file = os.path.join(image_dir, image_file)
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        labels, boxes, colors = [], [], []
        for anno in annos["labels"]:
            instance_id, category = anno["id"], anno["category"]
            box = anno["box2d"]
            box = [box["x1"], box["y1"], box["x2"], box["y2"]]
            labels.append(category + instance_id)
            boxes.append(box)
            if instance_id not in instance_color_dict:
                instance_color_dict[instance_id] = random_color(rgb=True, maximum=1)
            colors.append(instance_color_dict[instance_id])
        visualizer = Visualizer(image, metadata=None, scale=scale)
        vis = visualizer.overlay_instances(labels=labels, boxes=boxes, assigned_colors=colors)
        state = visualize(vis.get_image(), wait_time=wait)
        if state == VISUALIZE_EXIT:
            return


def bdd100k_mots_visualize(image_dir, anno_dir, wait=1, scale=1):
    assert image_dir and anno_dir
    image_list = sorted(os.listdir(image_dir))
    instance_color_dict = dict()
    for image_file in image_list:
        image_file = os.path.join(image_dir, image_file)
        anno_file = os.path.join(anno_dir, os.path.basename(image_file))[:-4] + ".png"
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        anno = cv2.imread(anno_file, cv2.IMREAD_UNCHANGED)
        # R = category id
        # G = 8 & truncated + 4 & occluded + 2 & crowd + ignore
        # B = ann_id for instance segmentation
        # A = ann_id for segmentation tracking
        counter = Counter(anno[..., 3].flatten())
        labels, masks, colors = [], [], []
        ignore = (anno[..., 1] & 0x1 == 0x1)
        crowd = (anno[..., 1] & 0x2 == 0x2)
        for key in counter.keys():
            if key == 0:
                continue
            mask = (anno[..., 3] == key)
            if (mask & ignore).any():
                key = "(I)" + str(key)
            if (mask & crowd).any():
                key = "(C)" + str(key)
            labels.append(key)
            masks.append(mask)
            if key not in instance_color_dict:
                instance_color_dict[key] = random_color(rgb=True, maximum=1)
            colors.append(instance_color_dict[key])
        visualizer = Visualizer(image, metadata=None, scale=scale)
        vis = visualizer.overlay_instances(labels=labels, masks=masks, assigned_colors=colors)
        state = visualize(vis.get_image(), wait_time=wait)
        if state == VISUALIZE_EXIT:
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Naive Visualizer")
    parser.add_argument("dataset", choices=[
        "kitti-mot", "kitti-mots", "apollo-ins", "apollo-mots",
        "bdd100k-mot", "bdd100k-mots"], help="dataset name")
    parser.add_argument("--image_dir", type=str, help="base directory of images")
    parser.add_argument("--anno_dir", type=str, help="base directory of annotations")
    parser.add_argument("--anno_file", type=str, help="annotation file for dataset")
    parser.add_argument("--base_dir", type=str, help="base directory for specific dataset like mot-challenge")
    parser.add_argument("--scale", type=float, default=1, help="image rescale to visualize")
    parser.add_argument("--wait", type=int, default=1, help="wait time for next frame in videos")
    parser.add_argument("--frame_rate", type=int, default=5, help="frame interval for visualization")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the samples")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    if args.shuffle:
        random.seed(args.seed)

    visualize_func_dict = {
        "kitti-mot": partial(kitti_mot_visualize, args.image_dir, args.anno_file, args.wait, args.scale),
        "kitti-mots": partial(kitti_mots_visualize, args.image_dir, args.anno_file, args.wait, args.scale),
        "apollo-ins": partial(apollo_ins_visualize, args.image_dir, args.anno_dir, args.scale, args.shuffle),
        "apollo-mots": partial(apollo_mots_visualize, args.image_dir, args.anno_dir, args.wait, args.scale),
        "bdd100k-mot": partial(bdd100k_mot_visualize, args.image_dir, args.anno_file, args.wait, args.scale),
        "bdd100k-mots": partial(bdd100k_mots_visualize, args.image_dir, args.anno_dir, args.wait, args.scale),
    }

    visualize_func_dict[args.dataset]()
