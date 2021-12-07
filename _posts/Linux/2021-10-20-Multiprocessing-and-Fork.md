---
date: 2021-10-20 14:24
title: "Multiprocessing and Fork"
categories: Fork
tags: Fork
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# multiprocessing in Windows
not working sinc


system fork
프로세스의 사본을 만드는게 fork 라고 한다.
자식을 만들기 위해 fork 하는 것이다.

linux는 가능하다. 하지만 윈도우에서는 불가능하다.

이와 같은 이유로 윈도우에서는 dataloader argument `num_workers = 0` 만 사용가능하다.

fork is insecure 하다.

<https://www.microsoft.com/en-us/research/uploads/prod/2019/04/fork-hotos19.pdf>


> multiprocessing not working in Windows: <https://purplechip.tistory.com/36>
> <https://medium.com/@grvsinghal/speed-up-your-python-code-using-multiprocessing-on-windows-and-jupyter-or-ipython-2714b49d6fac>  
> <https://www.pythonforthelab.com/blog/differences-between-multiprocessing-windows-and-linux/>