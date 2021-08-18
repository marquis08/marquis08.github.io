---
date: 2021-08-18 01:23
title: "Docker Container push"
categories: Docker
tags: Docker
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---


# Dockerhub에 Push 까지 하는 경우
private repository는 한개까지만 가능하고 더 쓰려면 유료 플랜에 가입해야 한다.

```sh
#docker commit <옵션> <컨테이너 이름> <이미지 이름>:<태그>
$ docker commit tf2 eungbean/tf2:latest

# login
$ docker login

# tag 지정 (생략가능)
# docker tag <이미지 이름>:<태그> <Docker 레지스트리 URL>/<이미지 이름>:<태그>
$ docker tag eungbean/deepo:latest

#push
# docker push <Docker 레지스트리 URL>/<이미지 이름>:<태그>
$ docker push eungbean/deepo:latest
```

# 로컬에서 commit 만
commit 까지만 하고 docker images 보면 이미지가 생성 되어있기는 하다.



> <https://eungbean.github.io/2018/12/03/til-docker-commit/>