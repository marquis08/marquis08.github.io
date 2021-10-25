---
date: 2021-09-27 14:30
title: "Bitbucket Pipeline"
categories: Bitbucket Pipeline
tags: Bitbucket Pipeline
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---


# SSH key trouble shooting
- --build-arg ssh_prv_key="$(cat ~/.ssh/id_rsa)" stargan

from repository setting
This private key will be added as a default identity in ~/.ssh/config.


- docker ARG 를 줘야지 --build-arg 에서 준 게 먹힌다. ssh private key 줄때 이렇게 해야된다.

원래목적:
- docker private registry 에 push 해서(네덜란드서버) docker compose할떄 해당 이미지를 받아서 실행

<https://support.atlassian.com/bitbucket-cloud/docs/build-and-push-a-docker-image-to-a-container-registry/>

- BITBUCKET_DOCKER_HOST_INTERNAL 를 사용해도 될듯 하다.
  -  <https://community.atlassian.com/t5/Bitbucket-articles/Changes-to-make-your-containers-more-secure-on-Bitbucket/ba-p/998464>
  -  <https://kevsoft.net/2021/07/29/connecting-to-service-containers-from-a-multi-staged-docker-build-in-bitbucket-pipeline.html>

# Appendix
- open addressing (충돌처리기법)
  - open hash, close hash