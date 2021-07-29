---
date: 2021-07-29 20:11
title: "Docker run through bind mount and assign name on container"
categories: DevCourse2 Docker
tags: DevCourse2 Docker
# 목차
toc: False  
toc_sticky: true 
toc_label : "Index"
---
- docker run with bind_mount and assign name to the container
```sh
docker run -d -v `pwd`:/whale_test -it --name docker123456 mydocker
```

- Assign a name to the container
```sh
docker run --name container_name
```



# Appendix
## Reference
> docker run: <https://docs.docker.com/engine/reference/commandline/run/>