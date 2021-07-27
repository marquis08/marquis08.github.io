---
date: 2021-07-28 03:08
title: "Docker Volume vs Bind Mount"
categories: DevCourse2 Docker
tags: DevCourse2 Docker
# 목차
toc: False  
toc_sticky: true 
toc_label : "Index"
---

👉 Reference: <https://www.daleseo.com/docker-volumes-bind-mounts/>  

# 볼륨 vs 바인드 마운트  
볼륨(volume)과 바인드 마운트(bind mount)의 가장 큰 차이점은 Docker가 해당 마운트 포인트를 관리해주느냐 안 해주느냐 입니다. 볼륨을 사용할 때는 우리가 스스로 볼륨을 생성하거나 삭제해야하는 불편함이 있지만, 해당 볼륨은 Docker 상에서 이미지(image)나 컨테이너(container), 네트워크(network)와 비슷한 방식으로 관리가 되는 이점이 있습니다. 그래서 **대부분의 상황에서는 볼륨을 사용하는 것이 권장되지만** **컨테이너화된 로컬 개발 환경을 구성할 때는 바인드 마운트가 더 유리**할 수 있습니다.  

로컬에서 개발을 할 때는 일반적으로 현재 작업 디렉터리에 프로젝트 저장소를 git clone 받아놓고 코드를 변경합니다. 따라서 바인드 마운트를 이용해서 해당 디렉터리를 컨테이너의 특정 경로에 마운트해주면 코드를 변경할 때 마다 변경 사항을 실시간으로 컨테이너를 통해 확인할 수 있습니다. 반대로 컨테이너를 통해 변경된 부분도 현재 작업 디렉터리에도 반영이 되기 때문에 편리할 것입니다.  

바인드 마운트를 실제 프로젝트에서 활용하는 예제는 [관련 포스트](https://www.daleseo.com/docker-nodejs/)를 참고 바라겠습니다.