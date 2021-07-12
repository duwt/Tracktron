## Tracktron

Tracktron is a multi-object tracking framework which extends [Detectron2](https://github.com/facebookresearch/detectron2) for supporting video input training and sequence inference. It is useful to build simultaneous detection and tracking algorithms on this framework. The evaluation part uses the implementation from [TrackEval](https://github.com/JonathonLuiten/TrackEval), which supports multiple metrics for multiple benchmarks. Now we have implemented [qdtrack](https://github.com/SysCV/qdtrack) on [bdd100k-mot](https://github.com/bdd100k/bdd100k) dataset as an example, which extends Mask R-CNN with an embed branch for object association.

### Results

| BDD100K MOT val | mMOTA | mIDF1 | mHOTA | MOTA | IDF1 | HOTA | mAP  | link                                                         |
| --------------- | ----- | ----- | ----- | ---- | ---- | ---- | ---- | ------------------------------------------------------------ |
| QDTrack(paper)  | 36.6  | 50.8  |       | 63.5 | 71.5 |      | 32.6 | [(code)](https://github.com/SysCV/qdtrack)                   |
| QDTrack(ours)   | 35.0  | 51.0  | 41.3  | 63.0 | 71.0 | 60.7 | 32.5 | [(model)](https://drive.google.com/file/d/12k8O1BpFz4AFFOEFNl2FH5kehLExHqgp/view?usp=sharing) |

### Install

Please refer to [INSTALL.md](INSTALL.md) for installation instructions.

### Usage

Most usages are inherited from Detectron2.

#### Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/qdtrack_bdd100k_mot.yaml \
    --num-gpus 4 OUTPUT_DIR output/qdtrack_bdd100k_mot
```

#### Inference

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/qdtrack_bdd100k_mot.yaml --eval-only \
    --num-gpus 4 OUTPUT_DIR output/qdtrack_bdd100k_mot \
    MODEL.WEIGHTS output/qdtrack_bdd100k_mot/model_final.pth
```

