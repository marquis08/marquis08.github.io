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

# DetectoRS 를 쓴 이유
instance-segmentation에서 그 당시 SOTA 였기 때문에  
<https://paperswithcode.com/sota/instance-segmentation-on-coco>  

지금은 `Dual-Swin-L(HTC, multi-scale)`


# Trouble Shooting
docker image에 mmdet이 없음..
- current version: mmdet-2.14.0
  - Got Error by AssertionError: MMCV==1.3.5 is used but incompatible. Please install mmcv>=1.3.8, <=1.4.0.
    - `pip install mmdet==2.13.0` [compatible-version-link](https://mmdetection.readthedocs.io/en/latest/get_started.html)
- KeyError: "HybridTaskCascade: 'DetectoRS_EfficientNet is not in the models registry'"
  - `mmcv/utils/registry.py` 에서 오류가 나는데..
  - 다른 모델로
- `detectors_resnest50.py` 로 다시
  - `pretrained` 에 wasabisys 를 쓰신거 보니 클라우드에 model weight 저장해 놓으신건가
- python 으로 mmdet import 하니까 incompatible 에러가 나서 (`AssertionError: MMCV==1.3.5 is used but incompatible. Please install mmcv>=1.1.5, <=1.3.`)
  - mmdet이랑 mmcv version을 맞춰줘야할 듯
    - 현재: `mmcv-full==1.3.5`, `mmdet==2.13.0`
    - 위에서 똑같은 에러가 나서 했는데 2.13.0 도 안되는 거였던것.
    - `dockerfile`에서 `mmcv-latest` 라고 되어있던 게 결국 문제발생.
    - 시도:
      - `pip install mmcv-full==1.2.7+torch1.6.0+cu102 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html`
      - dockerfile에도 이런식으로 지정해놔야 할듯.
      - mmcv releases: <https://github.com/open-mmlab/mmcv/releases>
    - 실패:
      - 또 incompatible 뜨네..
      - pip install mmdet==2.9.0
      - 


해결:  
- pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
- pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
- pip install mmdet==2.14.0
- 이렇게 install

문제:  
- run `default_runtime.py`
- `TypeError: __init__() missing 2 required positional arguments: 'doc' and 'pos'`
- 




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

