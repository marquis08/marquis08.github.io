---
date: 2021-08-19 00:52
title: "K-Fashion preprocessing"
categories: DevCourse2 FashionGAN_Proj
tags: DevCourse2 FashionGAN_Proj
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# K-Fashion Dataset Preprocessing To-do list
- json: coco format으로 json 만들어 줘야 함
- mv: data root 재설정
- rename: 파일이름에 공백과 괄호 등이 존재



## Data root 재설정
- mmdetection 에서는 `train/image/*`에 image 파일들이 존재하고 `train/anno/*` 에 json 파일들이 존재한다.
  - json은 전체로 묶인 파일 하나로 읽어오기 때문에 `anno`는 없어도 되었다.
- `train/image/*`와 `validation/image/*`로 압축 푼 이미지 들을 전체 옮겨 줘야 한다.
  - 다행히 json 에 파일이름은 파일이름 그 자체만 있기 때문에 coco로 만들어 줄때 처리를 안해줘도 되었다. (경로가 있을 가능성도 있었기 때문에)
- `shutil.move` 를 사용해서 src, dst 순서로 파일 이동

## json 파일에서 missing value가 있는 경우
- clean한 조건을 만족하는 image file name들을 필터링해서 list up 한 후, 해당 list에 있는 json 내용들만 다시 coco format의 json으로 재작업.

## Rename
- os.rename 과 re module 사용해서 rename function 을 사용

## K-Fashion to Coco
- deepfashion2 coco 를 참고해서 만들었다.
- rename fuction으로 json 에 rename된 파일명을 가지고 생성.




# Appendix
- Object Detection Task로 데이터셋을 구성할 때는 coco format을 기준으로 json을 만들어주고 파일명도 전처리를 따로 하지 않도록 해주는 것이 시간을 절약하는 효율적인 방법이다.
- Custom Datset을 만들때도 사용하고자 하는 library 의 format을 고려해서 만들어 주는 것이 애초에 시간을 절약하는 방법이 된다.


