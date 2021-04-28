---
date: 2021-04-29
title: "Cross Entropy"
categories: DevCourse2
tags: DevCourse2, Probability, LossFunction
# 목차
toc: False  
toc_sticky: true 
toc_label : "Index"
---

# Entropy

Cross Entropy를 다루기 전에 먼저 entropy에 대한 개념 정리를 해보자.

 

Entropy는 self-information의 평균이다.

 

self-information?

i(A) = log





## Pytorch CrossEntropy

파이토치 공식문서에 나온 예제

<br>

 

`python

import torch

import torch.nn as nn

loss = nn.CrossEntropyLoss()

data = torch.randn(3, 5, requires_grad=True)

target = torch.empty(3, dtype=torch.long).random_(5)

print(data)

print(target)

output = loss(data, target)

print(output)

output.backward()

`

 

`

tensor([[ 0.1449, -0.5543, -0.1170, -1.3969, -0.1700],

        [ 0.1715,  1.0151,  0.6917,  1.4723, -0.3305],

        [ 0.7153,  1.7428, -0.7265, -0.5458,  0.1957]], requires_grad=True)

tensor([2, 4, 1])

tensor(1.5741, grad_fn=<NllLossBackward>)

`

 

network를 돌고 나오면 class 개수 만큼 output으로 나온다.<br>

예제에서는 target으로 3개의 데이터만 주어진 것이다.<br>

data tensor에 있는 각 행마다 target의 각 element가 대응한다.

 

zero indexing이기 때문에 첫 행의 tensor는 3번째 데이터인 -0.1170을 가장 최대로 하는 값으로 업데이트 될 것이다.





<br>

 

- <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>