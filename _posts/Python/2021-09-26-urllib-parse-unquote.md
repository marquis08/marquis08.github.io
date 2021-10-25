---
date: 2021-09-26 14:30
title: "urllib parse unquote"
categories: urllib decode Python
tags: urllib decode Python
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---


> urllib.parse.unquote(string, encoding='utf-8', errors='replace')  
> Replace `%xx` escapes with their single-character equivalent. The optional encoding and errors parameters specify how to decode percent-encoded sequences into Unicode characters, as accepted by the bytes.decode() method.

```py
from urllib.parse import unquote
img_name = unquote(img_name)
# before decode:  %ED%95%A0%EB%8B%B9%EC%99%84%EB%A3%8C-t2804k8b(3)-%EC%9B%90%ED%8F%AC%EC%9D%B8%ED%8A%B8(%EC%83%98%ED%94%8C,%EB%A7%88%EA%B0%80%EB%A0%9BT)11.23-1-DSC_2674.jpg
# after decode:  할당완료-t2804k8b(3)-원포인트(샘플,마가렛T)11.23-1-DSC_2674.jpg
```




