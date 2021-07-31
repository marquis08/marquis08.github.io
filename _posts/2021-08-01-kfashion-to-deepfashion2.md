---
date: 2021-08-01 02:38
title: "Apply K-fashion 3rd solution to deepfashion2 dataset"
categories: DevCourse2 Docker
tags: DevCourse2 Docker
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Trouble Shooting
docker image에 mmdet이 없음..
- current version: mmdet-2.14.0
  - Got Error by AssertionError: MMCV==1.3.5 is used but incompatible. Please install mmcv>=1.3.8, <=1.4.0.
    - `pip install mmdet==2.13.0` [compatible-version-link](https://mmdetection.readthedocs.io/en/latest/get_started.html)
- KeyError: "HybridTaskCascade: 'DetectoRS_EfficientNet is not in the models registry'"
  - `mmcv/utils/registry.py` 에서 오류가 나는데..
  - 모델이 없는 건가.. 다른 backbone으로 해야되긴할듯..내일..


- preprocessing was done by `deepfashion2_to_coco.py` in deepfashion2.




created deepfasion2 in `mmdetection/config` (this is copy of k-fashion 3rd config and renamed it as deepfashion2)

- `/mmdetection/configs/deepfashion2`
  - `detectors_efficientnet_b0.py`
    - `./dataset.py` as \_base_\
      - change to `dataset_type = 'CocoDataset'`
      - change `data_root`
      - change transforms later on
      - data
        - data dict has: train_all, train, val, val_mini, test_val, test
          - what are those and where those data come from?
        - change `ann_file = data_root + 'train_json.json'`
        - change `img_prefix = data_root + 'train/image'`
      - evaluation
        - `metric=['proposal', 'bbox', 'segm']`
        - what is proposal and the the metric comes from?
    - `./schedule.py`
      - hyper parameters to change
        - optimizer, lr, lr_config
    - `./runtime.py`
      - changes
        - log_config - `interval`
        - workflow
          - `[('train', 1), ('val', 1)]`
    - `data`
      - `samples_per_gpu=4`
      - `workers_per_gpu=4`
    - Specify `work_dirs`

