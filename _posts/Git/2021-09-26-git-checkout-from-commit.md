---
date: 2021-09-26 14:30
title: "Git checkout from commit"
categories: Git submodule checkout
tags: Git submodule
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

특정 commit으로 부터 새로운 브랜치를 checkout 하는 cmd

```sh
git checkout -b new_branch commithash
```

commit hash는 git gui 툴로 보면 편하다. vscode 에서는 git graph extension을 사용중이다.