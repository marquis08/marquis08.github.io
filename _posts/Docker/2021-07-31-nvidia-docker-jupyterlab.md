---
date: 2021-07-31 16:33
title: "Docker gpu and jupyter lab"
categories: Docker
tags: Docker
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# 도커에서 Container 포트와 Host 포트의 개념


# if 8888 port is already in use
```sh
docker run -d -v `pwd`:/whale_test -it --name dev_jup --gpus all -p 8890:8890 --restart=always mydocker
```
and  
```sh
jupyter lab --allow-root --ip 0.0.0.0 --port=8890
```

done.  


# Summary
```sh
docker run -d -v `pwd`:/whale_test -it --name dev_jup --gpus all -p 8888:8888 --restart=always mydocker
```
```sh
docker exec -it container_id bash
```
```sh
conda install -c conda-forge jupyterlab
```
```sh
jupyter lab --allow-root --ip 0.0.0.0
```


# nvidia-docker (not necessarily)
```sh
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

```sh
sudo apt-get update
```

```sh
sudo apt-get install -y nvidia-docker2
```

```sh
sudo systemctl restart docker
```

```sh
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

# run Docker (with or without nvidia-docker)
```sh
nvidia-docker run -d -v `pwd`:/whale_test -it --name dev123 --gpus all -p 8888:8888 --restart=always mydocker
```

```sh
docker run -d -v `pwd`:/whale_test -it --name dev123 --gpus all -p 8888:8888 --restart=always mydocker
```

# Install jupyter lab in container
```sh
conda install -c conda-forge jupyterlab
```


# Jupyter lab 
```sh
jupyter lab --allow-root --ip 0.0.0.0
```




# Appendix
## Reference
> docker with pytorch and jupyter: <https://89douner.tistory.com/96>  
> nvidia-docker installation: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>  
> Docker 컨테이너 안에 jupyter 접속하기: <https://jybaek.tistory.com/812>  
> docker container with jupyter lab<https://anweh.tistory.com/68>  
> host port and container port: <https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=alice_k106&logNo=220278762795>