---
date: 2021-08-04 16:21
title: "Rename files with Python using os.rename"
categories: tmux
tags: tmux
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

```py
import os
 
def changeName(path, cName):
    i = 1
    for filename in os.listdir(path):
        print(path+filename, '=>', path+str(cName)+str(i)+'.jpg')
        os.rename(path+filename, path+str(cName)+str(i).zfill(6)+'.jpg')
        i += 1

changeName('real_images/','')
```


# Reference
> <https://data-make.tistory.com/171>