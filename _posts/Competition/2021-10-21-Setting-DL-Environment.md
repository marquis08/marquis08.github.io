---
date: 2021-10-21 00:24
title: "Setup DL Environment in Competition Container"
categories: Competition Pathology
tags: Competition Pathology
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

의료 대회에서 container로 진행하다보니  
필요한 파일들을 가상환경처럼 설치한 후 컨테이너 재시작을 하면 다 날아가 버린다.  
이에 따라 마운트 된 backup directory에 설치 및 저장 후 사용해야 하는데  

# pip install 경로지정 설치
pip install 로 저장 후 다시 설치된 라이브러리들이 제대로 호출 되는지 확인해야 한다.

`-t` argument를 줘서 경로 지정이 가능하다.

```sh
pip install tqdm -t ./backup/lib/tqdm/
```

해결: `./backup/lib` 에 필요한 모든 설치 패키지 설치 후

현재 디렉토리인 `/tf`에서 노트북 생성할때마다 sys path에 넣어줘야하는 번거로움이 있지만 아무튼 해결.

```py
import sys
sys.path.insert(0, './backup/lib')
```


# MMsegmentation error

에러메시지:  
`importerror: libtorch_cuda_cu.so: cannot open shared object file: no such file or directory`

cuda version 안맞아서 생기는 문제기 때문에 아래 명령어로 로 torch, cuda version 체크해주고 해당 버전에 맞는 패키지로 설치해주면 된다.
```py
import torch
print(torch.__version__, torch.cuda.is_available())
```



# pip uninstall 경로 지정
현재 그런 기능이 없기 때문에  

`pip install pckg -t your_dst`로 경로지정해서 설치한 후에는 해당 경로가서 디렉토리 삭제하는 방법으로만 가능하다고 현재는 알고 있다.



