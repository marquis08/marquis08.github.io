---
date: 2021-05-03 17:14
title: "Numpy & LinearAlgebra"
categories: DevCourse2
tags: DevCourse2 Numpy Latex Mathjax LinearAlgebra
# 목차
toc: True
toc_sticky: true 
toc_label : "Contents"
---

# Numpy

## Vector and Scalar
```python
import numpy
x = np.array([1,2,3])
c = 5
print("Addition: \t\t{}".format(x + c))
print("Subtraction: \t{}".format(x - c))
print("Multiplication: {}".format(x * c))
print("Division: \t\t{}\n".format(x / c))

### Output ###
# Addition:       [6 7 8]
# Subtraction:    [-4 -3 -2]
# Multiplication: [ 5 10 15]
# Division:       [0.2 0.4 0.6]
```
## Vector and Vector
$$y = \begin{pmatrix}1\\3\\5\end{pmatrix}$$  $$z = \begin{pmatrix}2\\9\\20\end{pmatrix}$$

```python
y = np.array([1,3,5])
z = np.array([2,9,20])
print("Addition: \t\t{}".format(y + z))
print("Subtraction: \t{}".format(y - z))
print("Multiplication: {}".format(y * z))
print("Division: \t\t{}".format(y / z))

### Output ###
# Addition:       [ 3 12 25]
# Subtraction:    [ -1  -6 -15]
# Multiplication: [  2  27 100]
# Division:       [0.5        0.33333333 0.25      ]
```

$$y = \begin{pmatrix}1\\3\\5\end{pmatrix}$$  $$z = \begin{pmatrix}2\\9\\20\end{pmatrix}$$ $$y+z = \begin{pmatrix}3\\12\\25\end{pmatrix}$$  
<br>

# Linear Algebra With Numpy
## Zero Vector(Matrix)
np.zeros(dim)
```python
np.zeros(3)
np.zeros((3,3,3))
```
## One Vector(Matrix)
np.ones(dim)
```python
np.ones(2)
np.ones((3,3))
```
## Diagonal Matrix
All zeros except main diagonal
np.diag((main_diagonal))
```python
np.diag((2,4))
np.diag((1,3,5))
```
## Identity Matrix
All zeros except main diagonal are ones
np.eye()
```python
np.eye(2, dtype=int) # n = 2 -> n*n
np.eye(3)
```
## Dot Product
np.dot() or @
```python
mat_1 = np.array([[1,4], [2,3]])
mat_2 = np.array([[7,9], [0,6]])

mat_1.dot(mat_2)
mat_1@mat_2
```
## Trace
The sum along diagonals of the array.  
np.trace()
```python
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr.trace()

np.eye(2, dtype=int).trace()
```
## Determinant
The determinant of an array.  
np.linalg.det()
```python
arr_2 = np.array([[2,3],[1,6]])
np.linalg.det(arr_2)

arr_3 = np.array([[1,4,7],[2,5,8],[3,6,9]])
np.linalg.det(arr_3)
```
$$det(A) = 0$$ : Col vectors are linearly dependent.
> If the determinant of a square matrix n×n A is zero, then A is not invertible. This is a crucial test that helps determine whether a square matrix is invertible, i.e., if the matrix has an inverse. When it does have an inverse, it allows us to find a unique solution, e.g., to the equation Ax=b given some vector b. When the determinant of a matrix is zero, the system of equations associated with it is **linearly dependent**; that is, if the determinant of a matrix is zero, at least one row of such a matrix is a scalar multiple of another.  

## Inversed Matrix
Inverse of a matrix.  
np.linalg.inv()
```python
mat = np.array([[1,4,],[2,3]])
mat_inv = np.linalg.inv(mat)

mat@mat_inv
```
## Eigenvalue and Eigenvector
정방행렬(n x n)에 대해서 $$Ax = \lambda x$$를 만족하는 $$\lambda$$와 x를 각각 고유값과 고유벡터라고 한다.  
np.linalg.eig()
```python
mat = np.array([[2,0,-2],[1,1,-2],[0,0,1]])

np.linalg.eig(mat)
eig_val, eig_vec = np.linalg.eig(mat)

mat @ eig_vec[:,0] # Ax
# Using matrix broadcasting not dot product for lambda x
eig_val[0] * eig_vec[:,0]# lambda x
```
## Getting L2 norm
np.linalg.norm(x, ord=2)
```python
import numpy as np
x = np.arange(9) - 4
print("Array: \t{}".format(x))
L2_norm = np.linalg.norm(x, axis=0, ord=2)
print("L2: \t{}".format(L2_norm))

### Output
# Array:  [-4 -3 -2 -1  0  1  2  3  4]
# L2:     7.745966692414834
```
## Getting Singular
Creating is_singular func.
```python
import numpy as np
arr_1 = np.array([[1,2],[3,6]])
arr_2 = np.array([[2,3],[1,6]])

def is_singular(arr):    
    det = np.linalg.det(arr)
    
    if not det:
        print("Singular\n")
    else:
        np.linalg.det(arr)
        print("Not Singular\ndet:\n\t{}".format(np.linalg.det(arr)))

is_singular(arr_1)
is_singular(arr_2)

### Output ###
# Singular

# Not Singular
# det:
#     9.000000000000002
```
<br>
---
### Latex Matrix representations
```Latex
pmatrix, bmatrix, vmatrix, Vmatrix are Latex environments:

    p for parentheses
    b for brackets
    v for verts
    B for braces
    V for double verts.
```

> numpy exercise: <https://www.machinelearningplus.com/python/101-numpy-exercises-python/>

> docs: <https://numpy.org/doc/stable/>

> Latex matrix: <https://www.math-linux.com/latex-26/faq/latex-faq/article/how-to-write-matrices-in-latex-matrix-pmatrix-bmatrix-vmatrix-vmatrix>

> Det is zero: <https://math.stackexchange.com/questions/355644/what-does-it-mean-to-have-a-determinant-equal-to-zero>