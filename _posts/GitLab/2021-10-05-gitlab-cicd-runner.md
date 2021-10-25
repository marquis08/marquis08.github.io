---
date: 2021-10-05 16:00
title: "GitLab cicd runner"
categories: GitLab cicd
tags: GitLab cicd
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# DIND

현재 yml 파일 기준으로 제일 위에 있는 image 가 현재 build될 이미지이름임

# Pytorch Docker image alpine
Alpine Linux 이미지 이기 때문에 apt-get 이 아니라
apt-get update 대신 apk update  
apt-get install 대신 apk add pkgname 을 사용해야 한다.


# Gitlab Runner in Docker Container (Docker in Docker setting)
1. Run gitlab runner inside Docker Container
<https://docs.gitlab.com/runner/install/docker.html>

make docker container for gitlab-runner

docker run 할때 `privileged` args를 줘야지 아래와 같은 에러가 나지 않는다.

또한 config.toml 안에도 privileged도 true를 줘야 함.

에러메시지: **docker: Cannot connect to the Docker daemon at tcp://docker:2375. Is the docker daemon running?**

해결방법:
```sh
docker run -d --name gitlab-runner --privileged \     
     -v /home/yilgukseo/DL/Vision/FridaGAN/gitlab-runner:/etc/gitlab-runner \
     -v /var/run/docker.sock:/var/run/docker.sock \
     gitlab/gitlab-runner:latest 
```

toml 파일이 아래 처럼되어야 하기 때문에 mount된 toml 파일에서 volumes에 기존에 없던 `"/certs/client"` 를 추가해주면 된다.

```toml
[[runners]]
  url = "https://gitlab.com/"
  token = TOKEN
  executor = "docker"
  [runners.docker]
    tls_verify = false
    image = "docker:19.03.12"
    privileged = true
    disable_cache = false
    volumes = ["/certs/client", "/cache"]
  [runners.cache]
    [runners.cache.s3]
    [runners.cache.gcs]
```
<https://docs.gitlab.com/ee/ci/docker/using_docker_build.html#docker-in-docker-with-tls-enabled>



2. register gitlab-runner inside docker container
`gitlab-runner register`

위 명령어로 gitlab-runner를 등록하면 다음과 같은 과정을 거친다.
- Enter the GitLab instance URL (for example, https://gitlab.com/):
    - https://gitlab.com/
- Enter the registration token:
    - your token from `settings/cicd/runners`
- Enter a description for the runner:
    - your desc
- Enter tags for the runner (comma-separated):
    - your tag
- Enter an executor: ssh, docker-ssh+machine, kubernetes, docker, parallels, shell, virtualbox, docker+machine, custom, docker-ssh:
    - docker
- Enter the default Docker image (for example, ruby:2.6):
    - docker:tagsyouwant


- message
Enter the GitLab instance URL (for example, https://gitlab.com/):
https://gitlab.com/
Enter the registration token:
{token}
Enter a description for the runner:
[dffbcd38570c]: jason3
Enter tags for the runner (comma-separated):
ceta3
Registering runner... succeeded                     runner=abcdefg
Enter an executor: ssh, docker-ssh+machine, kubernetes, docker, parallels, shell, virtualbox, docker+machine, custom, docker-ssh:
`docker`
Enter the default Docker image (for example, ruby:2.6):
`docker:latest`
Runner registered successfully. Feel free to start it, but if it's running already the config should be automatically reloaded! 

이 상태가 되고나서 web에 가서 `settings/cicd/runners` 에 가보면 초록불이 켜져 있어야 한다.  
다만 시간이 좀 걸리기 때문에 좀 기다리면 불이 켜져있음을 알 수 있다. 

# Docker in Docker network
기존의 unittest 구성방식 local 에서 진행하는 방식이어서 Docker in Docker (a.k.a. DinD) 환경이 아니었기 때문에 컨테이너간 접속이 가능했다.  
하지만 현재는 DinD 이기 때문에 현재 까지 알아본 바로는 도커 네트워크를 gitlab-runner 도커내부에서 구성해줘야 하기 때문에 골치가

# Docker image 맨뒤 args
기존 parse_args로 사용이 가능함.  

`docker run --net test_network --rm -e RABBITMQ_ENABLED -e RABBITMQ_HOST -e REDIS_ENABLED -e REDIS_HOST --name $CI_TEST_CONTAINER_NAME $CI_REGISTRY_IMAGE:$CI_COMMIT_TAG --debug --asset-url $CI_TEST_ASSET_URL_BASE --assets $CI_TEST_ASSET_0 --assets $CI_TEST_ASSET_1 --assets $CI_TEST_ASSET_2`

- ENTRYPOINT안에 bracket으로 감싸줘야하고 ""을 사용해야 args가 제대로 작동함.


# Appendix
## git mv

```sh
git mv oldname newname
```

is just shorthand for:


```sh
mv oldname newname
git add newname
git rm oldname
```

ref: <https://stackoverflow.com/questions/1094269/whats-the-purpose-of-git-mv>