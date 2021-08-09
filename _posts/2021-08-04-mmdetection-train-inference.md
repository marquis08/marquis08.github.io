---
date: 2021-08-04 15:25
title: "MMdetection Train & Inference"
categories: DevCourse2 ObjectDetection
tags: DevCourse2 ObjectDetection
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Train

```sh
python tools/train.py configs/_base_/deepfashion_dev1.py --work-dir working_test
```


# Inference to generate json file
```sh
python tools/test.py configs/_base_/deepfashion_dev1.py \ 
working_test/epoch_9.pth \
--format-only --eval-options "jsonfile_prefix=./result_epoch9"
```
<https://github.com/dacon-ai/K-fashion-baseline/blob/master/docs/getting_started.md>
<https://github.com/open-mmlab/mmdetection/blob/master/docs/tutorials/robustness_benchmarking.md>  


# Inference with shof resulting images

```sh
python tools/test.py configs/_base_/deepfashion_dev1.py \
 working_test/epoch_9.pth \
 --show-dir deepfashion_dev1_result
```

# Inference with existing models (input is only image)
<script src="https://gist.github.com/marquis08/3a471fbe96387e0fc114be00487d8b02.js"></script>

<https://github.com/open-mmlab/mmdetection/blob/master/demo/inference_demo.ipynb>