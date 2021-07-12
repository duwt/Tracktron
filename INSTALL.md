## Installation

You need to first install detectron2 according to its [documentation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html), and then run `python setup.py develop`.

## Prepare Dataset

This section shows how to convert detection or MOT dataset to Video COCO style, which is defined as follows. Items marked with * are added for tracking. (MOTS dataset is also supported and will be released later.)

```
- images [ ]
    - id
    - file_name
    - height
    - width
    - video_id*
    - video_name*
    - frame_id*
    - first_frame* (optional)
- annotations [ ]
    - id
    - bbox
    - area
    - segmentation
    - iscrowd
    - category_id
    - image_id
    - instance_id*
- categories [ ]
    - id
    - name
    - supercategory
```

### BDD100K MOT

```bash
ln -s /path/to/bdd100k datasets/bdd100k
python datasets/convert_dataset_to_coco_style.py bdd100k-mot \
    --anno_dir datasets/bdd100k/labels/box_track_20/train \
    --out_file datasets/bdd100k/bdd100k_box_track_20_train.json
python datasets/convert_dataset_to_coco_style.py bdd100k-mot \
    --anno_dir datasets/bdd100k/labels/box_track_20/val \
    --out_file datasets/bdd100k/bdd100k_box_track_20_val.json
```

### BDD100K Detection

```bash
ln -s /path/to/bdd100k datasets/bdd100k
python datasets/convert_dataset_to_coco_style.py bdd100k-det \
    --anno_file datasets/bdd100k/labels/det_20/det_train.json \
    --out_file datasets/bdd100k/bdd100k_det_mot_train.json
```
