---
date: 2021-09-26 14:30
title: "Git checkout submodule"
categories: Git
tags: Git submodule
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---


다른 branch로 checkout 할때 submodule 이 없을 경우 submodule init을 해주고 update --recursive를 해줘야 함.

```sh
git submodule init
git submodule update --recursive
```

하면 마지막으로 커밋했던 상태로 복원되기 때문에 여기서 checkout을 해주면 된다.


# Appendix
> official docs: <https://git-scm.com/book/ko/v2/Git-%EB%8F%84%EA%B5%AC-%EC%84%9C%EB%B8%8C%EB%AA%A8%EB%93%88>  
> <https://stackoverflow.com/questions/15124430/how-to-checkout-old-git-commit-including-all-submodules-recursively>  

