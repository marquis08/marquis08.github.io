---
date: 2021-08-18 01:09
title: "Docker Jupyter lab access denied"
categories: Docker
tags: Docker
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# 환경
- local에서 server에 있는 docker container에서 jupyter lab으로 접속

# 문제
솔직히 왜 에러가 났는지 모르겠다  
ip 주소가 바뀐 것 같다  
하지만 ip 주소 바꿔서 해봤는데도 적용이 안되길래 계속 서칭했으나..  

그래도 안되길래 그냥 혹시몰라서 아래 자료로 했는데 된다.  

# 결론
- stop running container 하고 재시작
- server docker container 가 접속이 안되면 container의 ip 주소 재확인
  - container ip 주소 확인: {%raw%}`docker inspect -f "{{ .NetworkSettings.IPAddress }}" CONTAINER_ID`{%endraw%}
- jupyter args에 container ip를 넣어준다
  - `jupyter lab --ip CONTAINER_IP --port PORT_NUM --allow-root`



> <https://opencvlib.weebly.com/blog/docker-jupyter-notebook>