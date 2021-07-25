---
date: 2021-07-25 16:18
title: "Docker install on ubuntu"
categories: Docker
tags: Docker
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Setup the docker via Repository
## 1. Update the apt package index and install packages to allow apt to use a repository over HTTPS:
```
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

Error: WARNING: unsafe ownership on homedir '/home/user/.gnupg'
```
$ sudo gpgconf --kill dirmngr
$ sudo chown -R $USER ~/.gnupg
```
## 2. Add Docker’s official GPG key:
```
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

## 3. Use the following command to set up the stable repository. To add the nightly or test repository, add the word nightly or test (or both) after the word stable in the commands below. Learn about nightly and test channels.
```
$ echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

# Install Docker Engine
## 1. Update the apt package index, and install the latest version of Docker Engine and containerd, or go to the next step to install a specific version:
```
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
```

## 2. Verify that Docker Engine is installed correctly by running the hello-world image.
```
$ sudo docker run hello-world
```

```md
Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

# Docker add & remove user authorization
## add
```
$ sudo usermod -aG docker [your-user]
```  
## Remove
```
$ sudo deluser [your-user] docker 
```

# Remove Docker 
```
# 도커 패키지 제거
$ sudo apt-get purge docker-ce

# 이미지, 컨테이너, 볼륨, 사용자 지정 설정 파일은 패키지 제거로 제거 되지 않음. 별도 제거 필요함.
$ sudo rm -rf /var/lib/docker
```



# Docker Software
- 도커 엔진(Docker Engine): 특정한 소프트웨어를 도커 컨테이너로 만들고 실행하게 해주는 데몬(Daemon)을 의미합니다. 도커 엔진과 도커 클라이언트 사이에는 REST API가 사용됩니다. REST API 서버에 요청을 보내는 것이 도커 클라이언트입니다.

- 도커 클라이언트(Docker Client): 도커 엔진과 통신하는 소프트웨어로 개발자가 직접 이용할 수 있습니다. 윈도우(Window)/맥(Mac)/리눅스(Linux)를 지원합니다. 물론 윈도우는 도커를 사용하기에 최악의 조건이고, 리눅스가 제일 최상의 조건이지만 도커는 공식적으로 윈도우를 제대로 지원하고 있습니다.

- 도커 호스트 운영체제(Docker Host OS): 도커 엔진을 설치할 수 있는 운영체제 환경을 의미합니다. 64비트 리눅스 커널 버전 3.10 이상 환경을 의미하고, 32비트 환경에서는 도커 엔진이 돌아가지 않는답니다. 애초에 초기의 도커 이미지는 심지어 오직 우분투(Ubuntu) 운영체제 전용이었어요. 현재는 우분투, CentOS, Debian, Fedora 등에서 사용할 수 있게 되었지만요.

- 도커 머신(Docker Machine): 로컬 및 원격지 서버에 도커 엔진을 설치하고, 다양한 환경 설정을 자동으로 수행해주는 클라이언트를 의미합니다.


# Docker commands
- `docker ps`: 현재 돌아가고 있는 컨테이너를 확인하는 명령어입니다.

- `docker images`: 현재 도커 머신에 설치된 도커 이미지를 확인하는 명령어입니다.

- `docker --version`: 도커 버전확인

- `sudo docker info`: 상세한 버전 확인

- `docker run hello-world`: 도커에서 헬로우 월드(Hello World)를 띄우는 명령어입니다. 우리 눈에는 보이지 않지만 다음의 과정이 포함되는 겁니다.
    - 공식 사이트에서 install 할때 출력해봤음.  

- `docker images`: 현재 도커 머신에 존재하는 이미지 목록을 출력합니다.
    ```
    (base)  yilgukseo  ~/DL  sudo docker images
    REPOSITORY    TAG       IMAGE ID       CREATED        SIZE
    hello-world   latest    d1165f221234   4 months ago   13.3kB
    ```  

- `docker ps -a`: 현재 도커 머신에 존재하는 컨테이너를 출력합니다.
    ```
    (base)  yilgukseo  ~/DL  sudo docker ps -a
    CONTAINER ID   IMAGE         COMMAND    CREATED         STATUS                     PORTS     NAMES
    fad8e07c603f   hello-world   "/hello"   9 minutes ago   Exited (0) 9 minutes ago             eloquent_poincare
    ```
    - ID, IMAGE, Command, CREATED, STATUS, PORTS, NAMES 를 확인

- `docker rm (컨테이너 ID)`: 특정 컨테이너를 삭제합니다.

- `docker rm (컨테이너 ID) -f`: 특정 컨테이너를 강제로 삭제합니다.

- `docker run -it python`: 파이썬 이미지를 다운로드 받아서 실행해주는 명령어입니다.
  - `-it` 옵션은 표준 입출력을 이용해 컨테이너에 명령어를 입력할 수 있게 해줍니다.
    ```
    (base)  yilgukseo  ~/DL  sudo docker run -it python
    Unable to find image 'python:latest' locally
    latest: Pulling from library/python
    627b765e08d1: Pull complete 
    c040670e5e55: Pull complete 
    073a180f4992: Pull complete 
    bf76209566d0: Pull complete 
    ca7044ed766e: Pull complete 
    7b16520e0e66: Pull complete 
    e121e5a178df: Pull complete 
    abbaf10bd160: Pull complete 
    4349f8e0b43a: Pull complete 
    Digest: sha256:a465eb577326845a6772c0de37d11c4de3da9a0248c85ae8a7b8629561cb2185
    Status: Downloaded newer image for python:latest
    Python 3.9.6 (default, Jul 22 2021, 15:16:20) 
    [GCC 8.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> print("hellow docker python")
    hellow docker python
    ```

- `quit()`: 파이썬 콘솔을 종료시키는 명령어입니다.

- `docker start (컨테이너 ID)`: 종료된 컨테이너를 재시작하는 명령어입니다.

- `docker attach (컨테이너 ID)`: 특정 컨테이너로 재접속하는 명령어입니다.

- `docker exec -it (컨테이너 ID) bash`: 실행 중인 컨테이너에 배시(Bash) 쉘로 접속하는 명령어입니다.

- `sudo docker stop container_id`: stop container
    ```
    (base)  ✘ yilgukseo  ~/DL  sudo docker ps -a
    CONTAINER ID   IMAGE         COMMAND     CREATED          STATUS                      PORTS     NAMES
    0a645f33513a   python        "python3"   11 minutes ago   Up About a minute                     busy_wescoff
    fad8e07c603f   hello-world   "/hello"    23 minutes ago   Exited (0) 23 minutes ago             eloquent_poincare
    (base)  yilgukseo  ~/DL  sudo docker stop 0a645f33513a
    0a645f33513a
    (base)  yilgukseo  ~/DL  sudo docker ps -a            
    CONTAINER ID   IMAGE         COMMAND     CREATED          STATUS                        PORTS     NAMES
    0a645f33513a   python        "python3"   15 minutes ago   Exited (137) 54 seconds ago             busy_wescoff
    fad8e07c603f   hello-world   "/hello"    27 minutes ago   Exited (0) 27 minutes ago               eloquent_poincare
    ```
    - stop 하면 `Exited` 라고 표시됨.
      - start는 빨리 되는데 Stop 은 시간이 좀 더 소요됨
    - Container name으로도 되는지 확인
      - 됨.
        ```
        (base)  yilgukseo  ~/DL  sudo docker ps -a             
        CONTAINER ID   IMAGE         COMMAND     CREATED          STATUS                      PORTS     NAMES
        0a645f33513a   python        "python3"   16 minutes ago   Up 1 second                           busy_wescoff
        fad8e07c603f   hello-world   "/hello"    28 minutes ago   Exited (0) 28 minutes ago             eloquent_poincare
        (base)  yilgukseo  ~/DL  sudo docker stop busy_wescoff
        busy_wescoff
        (base)  yilgukseo  ~/DL  sudo docker ps -a            
        CONTAINER ID   IMAGE         COMMAND     CREATED          STATUS                        PORTS     NAMES
        0a645f33513a   python        "python3"   17 minutes ago   Exited (137) 15 seconds ago             busy_wescoff
        fad8e07c603f   hello-world   "/hello"    29 minutes ago   Exited (0) 28 minutes ago               eloquent_poincare
        ```

- `docker rm [-option] [container ID]`: 컨테이너 삭제

- `docker rm $(docker ps -a -q -f status=exited)`: 중지된 컨테이너 삭제

- `docker rm $(docker ps -qa)`: 모든 컨테이너 삭제

official docker commands: <https://docs.docker.com/engine/reference/commandline/docker/>  

# Appendix
## Reference
> Official site: <https://docs.docker.com/engine/install/ubuntu/>  
> <https://shaul1991.medium.com/%EC%B4%88%EB%B3%B4%EA%B0%9C%EB%B0%9C%EC%9E%90-%EC%9D%BC%EC%A7%80-docker-%EC%84%A4%EC%B9%98%ED%95%B4%EB%B3%B4%EC%9E%90-d3d6a11ea098>  
> <https://unix.stackexchange.com/questions/452020/gpg-warning-unsafe-ownership-on-homedir-home-user-gnupg>  
> ndb : <https://ndb796.tistory.com/91>  