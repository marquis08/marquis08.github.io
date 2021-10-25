---
date: 2021-04-30
title: "Numpy Array Broadcasting"
categories: DevCourse2 DevCourse2_DL_Math
tags: DevCourse2 Numpy
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

# Numpy Array Broadcasting

Docs says:
> The term broadcasting describes how numpy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes.  

<br>
array가 다른 shape일때, 특정 제약조건이 있긴하지만,<br>
더 작은 array가 더 큰 array 전반에 걸쳐 "broadcast" 된다는 것.  

<br>Example from docs
```python
import numpy as np
a = np.array([0.0, 10.0, 20.0, 30.0])
b = np.array([1.0, 2.0, 3.0])
c = a[:, np.newaxis] + b
print(a)
print("\nnp.newaxis: \n",a[:, np.newaxis])
print("newaxis shape:  ",a[:, np.newaxis].shape)
print("b shape:   \t\t",b.shape,"\n")
print(a[:, np.newaxis] + b)
print("broadcasted shape: ",c.shape)

######### output #########
# [ 0. 10. 20. 30.]

# np.newaxis: 
#  [[ 0.]
#  [10.]
#  [20.]
#  [30.]]
# newaxis shape:   (4, 1)
# b shape:         (3,) 

# [[ 1.  2.  3.]
#  [11. 12. 13.]
#  [21. 22. 23.]
#  [31. 32. 33.]]
# broadcasted shape:  (4, 3)
```


Example of Addition: add +1 to 2nd row
```python
import numpy as np
a = np.arange(1,10).reshape(3,3)
b = np.array(
    [[0,0,0],
    [1,1,1],
    [0,0,0]])
print(a,"\n")
print(b,"\n")
print(a+b,"\n")

#### OUTPUT ####
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]] 

# [[0 0 0]
#  [1 1 1]
#  [0 0 0]] 

# [[1 2 3]
#  [5 6 7]
#  [7 8 9]] 
```

Example of Multiplication: not matrix multiplication
```python
import numpy as np
a = np.arange(1,10).reshape(3,3)
b = np.array(
    [[0,1,-1],
    [0,1,-1],
    [0,1,-1]])
print(a,"\n")
print(b,"\n")
print(a*b,"\n")

# [[1 2 3]
#  [4 5 6]
#  [7 8 9]] 

# [[ 0  1 -1]
#  [ 0  1 -1]
#  [ 0  1 -1]] 

# [[ 0  2 -3]
#  [ 0  5 -6]
#  [ 0  8 -9]] 

```

### Then, Matrix multiplication ?
```python
print(a.dot(b))
# [[  0   6  -6]
#  [  0  15 -15]
#  [  0  24 -24]]
```

## np.dot VS. np.matmul
np.dot()

: 만약 a가 N차원 배열이고 b가 2이상의 M차원 배열이라면, dot(a,b)는 a의 마지막 축과 b의 뒤에서 두번째 축과의 내적으로 계산된다.

np.matmul()

: 만약 배열이 2차원보다 클 경우, 마지막 2개의 축으로 이루어진 행렬을 나머지 축에 따라 쌓아놓은 것이라고 생각한다.


출처: https://ebbnflow.tistory.com/159 [Dev Log : 삶은 확률의 구름]



<br>

> numpy docs:  <https://numpy.org/doc/stable/user/basics.broadcasting.html><br>
> np.dot VS. np.matmul: <https://ebbnflow.tistory.com/159>