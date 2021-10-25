---
date: 2021-08-25 01:20
title: "Coco Annotator install and usage"
categories: DevCourse2 Docker CocoAnnotator FashionGAN_Proj
tags: DevCourse2 Docker CocoAnnotator FashionGAN_Proj
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Coco Annotator 설치

docker & docker compose 설치 후

```sh
git clone https://github.com/jsbroks/coco-annotator.git

cd coco-annotator
```

# Installing - Production Build
```sh
docker-compose up 
```

후

`localhost:5000` 하면 실행된다.  

coco-annotator dir의  dataset dir에 dir과 파일을 넣어주면 dataset이 생성된 것을 볼 수 있다.  

카테고리는 위의 nav bar에서 카테고리르 먼저 등록해줘야 dataset에서 카테고리 선택이 가능하다.  

# Stop Container
docker-compose down


> <https://min0114.tistory.com/18>  
> <https://github.com/jsbroks/coco-annotator>  
> <https://github.com/jsbroks/coco-annotator/wiki/Getting-Started#Prerequisites>  