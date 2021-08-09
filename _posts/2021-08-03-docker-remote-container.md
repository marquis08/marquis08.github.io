---
date: 2021-08-03 15:54
title: "Docker Remote-Containers: Attach to running container"
categories: Docker
tags: Docker
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

서버에 있는 docker container에 vscode로 접속하는 방법임 

접속은 되는데 vscode explorer를 사용해야 편리하기 때문에 접속 경로를 바꿔줘야 편리함


remote-containers: open container configuration file 을 열어보니  
json 으로 config가 되어있고
현재 docker의 docker_image_name.json 형태로 config가 되어있음

아래와 같이:  
```json
{
	"workspaceFolder": "/root"
}
```

이걸 `/root` 에서 `/`로 바꾸면 explorer 가 잘 작동할거 같다.  

remote-ssh 같은 경우에는 open folder로 다시 해당 folder로 접속할 수 있는데  
remote-container는 없는 듯.  

성공👋

근데 생각보다 로딩이 느리다는 점 말고는 좋다.
