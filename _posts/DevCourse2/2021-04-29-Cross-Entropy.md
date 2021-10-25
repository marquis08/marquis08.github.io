---
date: 2021-04-29
title: "Cross Entropy"
categories: DevCourse2 DevCourse2_DL_Math
tags: DevCourse2 Probability LossFunction
# 목차
toc: False  
toc_sticky: true 
toc_label : "Index"
---

# Entropy

Entropy는 self-information의 평균

What is self-information?

> Within the context of information theory, self-information is defined as the amount of information that knowledge about (the outcome of) a certain event, adds to someone's overall knowledge. The amount of self-information is expressed in the unit of information: a bit. <br> <https://psychology.wikia.org/wiki/Self-information>

> 정보이론의 핵심 아이디어는 잘 일어나지 않는 사건(unlikely event)은 자주 발생하는 사건보다 정보량이 많다(informative)는 것입니다. 예컨대 ‘아침에 해가 뜬다’는 메세지로 보낼 필요가 없을 정도로 정보 가치가 없습니다. 그러나 ‘오늘 아침에 일식이 있었다’는 메세지는 정보량 측면에서 매우 중요한 사건입니다. 이 아이디어를 공식화해서 표현하면 다음과 같습니다.
- 자주 발생하는 사건은 낮은 정보량을 가진다. 발생이 보장된 사건은 그 내용에 상관없이 전혀 정보가 없다는 걸 뜻한다.
- 덜 자주 발생하는 사건은 더 높은 정보량을 가진다.
- 독립사건(independent event)은 추가적인 정보량(additive information)을 가진다. 예컨대 동전을 던져 앞면이 두번 나오는 사건에 대한 정보량은 동전을 던져 앞면이 한번 나오는 정보량의 두 배이다.
> <https://ratsgo.github.io/statistics/2017/09/22/information/>


각 사건에 대해서 정보의 양(확률변수)에 대한 기댓값이라고 볼 수 있다.

$$ i(A) =  log_b({1\over P(A)})  = -log_b{P(A)} $$

b: 정보의 단위 보통 e를 사용한다.

Entropy:  


$$H(X) = \sum_j{P(A_j)i(A_j)} = -\sum{P(A_j)log_2P(A_j)}$$

---
# Cross Entropy


Cross Entropy는 Deep Learning에서 주로 Loss Function으로 사용된다.<br>
다시 확률에서의 CrossEntropy를 보자.

P와 Q라는 확률분포가 존재할 때의 크로스 엔트로피는 어떤 의미인가.

크로스 엔트로피 (P, Q):  
Q의 상황에서의 자기정보(A)가 실제 P 상에서 P의 확률로 나타냈을 때의 기댓값.

Cross Entropy를 사용해서 P와 Q의 확률분포를 비교한다.  
P는 ground truth라고 보면 되고, Q는 예측값이라고 보면 된다.  

보통 DL을 할때 label은 ohe를 안해도 output과 label을 입력하면,  
pytorch CrossEntropy 클래스에서 자동으로 해결이 가능하다.


---
파이토치 기준으로는 BinaryCrossEntropy, CrossEntropy 클래스가 존재한다.

### Pytorch CrossEntropy
<br>
파이토치 공식문서에 나온 예제<br>

```python

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
```
output

```
tensor([[ 0.1449, -0.5543, -0.1170, -1.3969, -0.1700],
        [ 0.1715,  1.0151,  0.6917,  1.4723, -0.3305],
        [ 0.7153,  1.7428, -0.7265, -0.5458,  0.1957]], requires_grad=True)

tensor([2, 4, 1])

tensor(1.5741, grad_fn=<NllLossBackward>)
```

network를 돌고 나오면 class 개수 만큼 output으로 나온다.<br>
예제에서는 target으로 3개의 데이터만 주어진 것이다.<br>
data tensor에 있는 각 행마다 target의 각 element가 대응한다.<br>
zero indexing이기 때문에 첫 행의 tensor는 3번째 데이터인 -0.1170을 가장 최대로 하는 값으로 업데이트 될 것이다.

> <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>