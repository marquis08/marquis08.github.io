---
date: 2021-09-26 14:30
title: "Docker Redis"
categories: Docker Redis
tags: Docker Redis
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Redis with Docker
- get redis image from docker hub
  -  
    ```sh
    Docker image pull redis
    ```
- redis-cli와 같이 구동해야 됨. 두 컨테이너를 연결하기 위해 docker network 구성해야 한다.
  - create network
    ```sh
    docker network create redis-net
    ```
 -  get network list
    ```sh
    docker network ls
    ```
- docker run, volume을 저장하지 않고 사용하는 명령어
```sh
docker run --name redis001 -p 6379:6379 --network redis-net -d redis:6.2.5 redis-server --appendonly yes 
```

# Docker Network
> Docker 컨테이너(container)는 격리된 환경에서 돌아가기 때문에 기본적으로 다른 컨테이너와의 통신이 불가능합니다. 하지만 여러 개의 컨테이너를 하나의 Docker 네트워크(network)에 연결시키면 서로 통신이 가능해집니다. 이번 포스팅에서는 컨테이너 간 네트워킹이 가능하도록 도와주는 Docker 네트워크에 대해서 알아보도록 하겠습니다.

network prune
```sh
docker network prune
```


# Appendix
## Reference
> <https://jistol.github.io/docker/2017/09/01/docker-redis/how-to-checkout-old-git-commit-including-all-submodules-recursively>  
>  <https://emflant.tistory.com/235>  
>  <https://littleshark.tistory.com/68>  
> <https://www.daleseo.com/docker-networks/>

