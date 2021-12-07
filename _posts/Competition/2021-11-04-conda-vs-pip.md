---
date: 2021-11-04 04:20
title: "Conda Vs. Pip"
categories: Competition Pathology Conda Segmentation
tags: Competition Pathology Conda Pip Segmentation
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

conda 로 환경 만들어서 torch install을 conda로 했더니 cuda False가 되버려서 버전 문제인줄 알았더니 pip install 로 하니 문제 없이 cuda 사용 가능하게 되었다.  

conda pckg로 사용하는 것과 pip로 사용하는 것에는 어떤 차이가 존재할까.  

- Pip: Python libraries only
- Conda: Any dependency can be a Conda package (almost)

![conda_vs_pip](https://i.stack.imgur.com/1UKJt.png)
# Appendix
## Reference
> <https://pythonspeed.com/articles/conda-vs-pip/>
> 
> <https://stackoverflow.com/questions/20994716/what-is-the-difference-between-pip-and-conda>