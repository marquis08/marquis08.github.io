---
date: 2021-11-01 14:39
title: "Setup Env with conda"
categories: Competition NoonBody Segmentation
tags: Competition NoonBody Segmentation
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

대회마다 패키지 버전 이슈가 있을 수 있기 때문에 대회 참여할 때마다 다른 conda 환경을 만들어주고 시작하는게 좋다고 생각한다.

특정 패키지의 경우 dependent 패키지들의 버전에 따라 설치가 되기 때문에 (mmdetection)

kaggle 노트북 환경을 yaml로 받아오거나 requirements로 받아서 실행할 경우 아래의 명령어를 통해 conda env create가 가능하다.

```sh
conda create --n myenv --file environment.yaml
```

혹은 `conda create` 으로 환경을 만든 후에 `pip install -r requirements.txt` 명령어를 통해 필요한 패키지들을 설치해 줄 수 있다.  

# Appendix
## Reference
> <https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf>

