---
date: 2021-05-07 03:49
title: "Git Large File push Error and Solution"
categories: DevCourse2
tags: Git Github
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

Plotting 과제 push하다가 100MB 넘는 파일을 push하려고 하니 error가 났다.  
100MB 이상 파일 올리려면 Git LFS(Large File Storage)를 사용해야 한다는데  
굳이 올리지 않아도 된다면, 에러만 해결하고 다시 PUSH 하고 싶다면  
아래 링크를 참조해서  
commit을 취소하는 방법을 사용하면 된다.  
이렇게 이전 상태로 다시 되돌린 다음에 해당 파일을 지우고  

다시 add commit push 순으로 하면 된다.

```terminal
git reset --hard HEAD^
```

거의 1시간 날린듯  🤮🤮🤮


> git 취소하기 <https://velog.io/@hidaehyunlee/Git-add-commit-push-%EC%B7%A8%EC%86%8C%ED%95%98%EA%B8%B0>