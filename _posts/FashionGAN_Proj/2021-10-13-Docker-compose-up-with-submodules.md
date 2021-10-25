---
date: 2021-10-13 17:44
title: "FGAN Proj - Docker compose up with submodules as CICD"
categories: DevCourse2 Docker CICD FashionGAN_Proj Gitlab
tags: DevCourse2 Docker CICD FashionGAN_Proj Gitlab
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# clone proj from gitlab

use `--recursive` to automatically initialize and update each submodule in the repository, including nested submodules if any of the submodules in the repository have submodules themselves.

```sh
git clone --recursive git@gitlab.com:frida129/test_compose.git 
```
# Test Local docker compose befor use gitlab runner





>submodule docs: <https://git-scm.com/book/en/v2/Git-Tools-Submodules>