---
date: 2021-07-27 23:40
title: "Docker Build"
categories: DevCourse2 Docker
tags: DevCourse2 Docker
# 목차
toc: False  
toc_sticky: true 
toc_label : "Index"
---

# Build Docker Image, run container, start container, and stop container
- by using shell command
```sh
sudo bash docker/build.sh
```

- or by using dockerfile directly
```sh
sudo docker build Dockerfile
```

- After built, run docker.
```sh
sudo docker run tagged_name
```

- start container
```sh
sudo docker container start container_id
```

- access to /bin/bash
```
sudo docker exec -it container_id \bin\bash
```

- stop docker container
```
sudo docker stop container_id
```


# Keep running Docker container
- add `ENTRYPOINT` in `Dockerfile` as last line.
```dockerfile
ENTRYPOINT ["tail", "-f", "/dev/null"]
```

> ref: <https://devopscube.com/keep-docker-container-running/>

## Docker `ENTRYPOINT` Instruction
`ENTRYPOINT ["tail", "-f", "/dev/null"]` 로 해야 keep running.  

tail : 맨뒤에 다섯줄 보여줌  
-f : 피드를 계속 읽어줌  
/dev/null: null 을 계속 읽어줌  


# For the future work.
Never write `latest` on the file name in any enviroment related files.  

Write very specific version on every files in use for compatibility.  







