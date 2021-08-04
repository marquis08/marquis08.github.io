---
date: 2021-08-03 20:28
title: "Docker Container memory shortage"
categories: NLP
tags: NLP
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

Error Message:  
`RuntimeError: DataLoader worker (pid 4301) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.`  

해결책으로는 


--ipc 혹은 --shm-size를 사용해서 docker run 하면됨

ipc에 대해서 잠깐 검색해봤는데 잠깐으로는 부족할 것 같아서

일단  

1. docker run --ipc=host
1. docker run --shm-size=64G

정도의 해결 책이 제시되어 있었다.

```sh
docker run -d -v `pwd`:/proj -it --name dev_dpfash3 --gpus all -p 8891:8891 --restart=always mydocker --shm-size=8G
```

# Reference
> docker 컨테이너에서 pytorch 실행시 메모리 에러 해결: <https://curioso365.tistory.com/136>  
> [Trouble Shooting] 도커 사용시 문제 발생 및 해결: <https://soyoung-new-challenge.tistory.com/70>  
> kubernetes shared memory 사이즈 조정: <https://ykarma1996.tistory.com/106>  