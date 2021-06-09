---
date: 2021-06-10 02:30
title: "ML basics - Linear Algebra"
categories: DevCourse2 LinearAlgebra MathJax
tags: DevCourse2 LinearAlgebra MathJax
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# 왜 선형대수를 알아야 하는가?

Deep learning을 이해하기 위해서 반드시 선형대수 + 행렬미분 + 확률의 탄탄한 기초가 필요하다.

예) Transformer의 attention matrix:  

$$\mathrm{Att}_{\leftrightarrow}(Q, K, V) = D^{-1}AV, {\sim}A = \exp(QK^T/\sqrt{d}), {\sim}D = \mathrm{diag}(A1_L)$$

이렇게 핵심 아이디어가 행렬에 관한 식으로 표현되는 경우가 많다.  
> 목표: 선형대수와 행렬미분의 기초를 배우고 간단한 머신러닝 알고리즘(PCA)을 유도해보고자 한다.  


# 기본 표기법 (Basic Notation)

- $$A\in \mathbb{R}^{m\times n}$$는 $$m$$개의 행과 $$n$$개의 열을 가진 행렬을 의미한다.
- $$x \in \mathbb{R}^n$$는 $$n$$개의 원소를 가진 벡터를 의미한다. $$n$$차원 벡터는 $$n$$개의 행과 1개의 열을 가진 행렬로 생각할 수도 있다. 이것을 열벡터(column vector)로 부르기도 한다. 만약, 명시적으로 행벡터(row vector)를 표현하고자 한다면, $$x^T$$($$T$$는 transpose를 의미)로 쓴다.
- 벡터 $$x$$의 $$i$$번째 원소는 $$x_i$$로 표시한다.  

$$\begin{align*} x = \begin{bmatrix} x_1\\ x_2\\ \vdots\\ x_n \end{bmatrix} \end{align*}$$  

- $$a_{ij}$$(또는 $$A_{ij}, A_{i,j}$$)는 행렬 $$A$$의 $$i$$번째 행, $$j$$번째 열에 있는 원소를 표시한다.  

$$\begin{align*} A = \begin{bmatrix} a_{11} &\ a_{12} &\ \cdots &\ a_{1n}\\ a_{21} &\ a_{22} &\ \cdots &\ a_{2n}\\ \vdots &\ \vdots &\ \ddots &\ \vdots\\ a_{m1} &\ a_{m2} &\ \cdots &\ a_{mn} \end{bmatrix} \end{align*}$$  

- $$A$$의 $$j$$번째 열을 $$a_j$$ 혹은 $$A_{:,j}$$로 표시한다.  

$$\begin{align*} A = \begin{bmatrix} \vert &\ \vert &\ &\ \vert\\ a_1 &\ a_2 &\ \cdots &\ a_n\\ \vert &\ \vert &\ &\ \vert \end{bmatrix} \end{align*}$$  

- $$A$$의 $$i$$번째 행을 $$a_i^T$$ 혹은 $$A_{i,:}$$로 표시한다.  

$$\begin{align*} A = \begin{bmatrix} - & a_1^T & -\\ - & a_2^T & -\\ & \vdots &\\ - & a_m^T & - \end{bmatrix} \end{align*}$$  




Python에서의 벡터, 행렬 표현방법:  
```python
In [1]: [10.5, 5.2, 3.25, 7.0]
Out[1]: [10.5, 5.2, 3.25, 7.0]

In [2]: import numpy as np
        x = np.array([10.5, 5.2, 3.25])

In [3]: x.shape
Out[3]: (3,)

In [4]: i = 2
        x[i]
Out[4]: 3.25

In [5]: np.expand_dims(x, axis=1).shape
Out[5]: (3, 1)

In [6]: A = np.array([
            [10,20,30],
            [40,50,60]
        ])
        A
Out[6]: array([[10, 20, 30],
             [40, 50, 60]])

In [7]: A.shape
Out[7]: (2, 3)

In [8]: i = 0
        j = 2
        A[i, j]
Out[8]: 30

In [9]: # column vector
        j = 1
        A[:, j]
Out[9]: array([20, 50])

In [10]:# row vector
        i = 1
        A[i, :]
Out[10]:array([40, 50, 60])
```  


## 행렬의 곱셉 (Matrix Multiplication)

두 개의 행렬 $$A\in \mathbb{R}^{m\times n}$$, $$B\in \mathbb{R}^{n\times p}$$의 곱 $$C = AB \in \mathbb{R}^{m\times p}$$는 다음과 같이 정의된다.  

$$C_{ij} = \sum_{k=1}^n A_{ik}B_{kj}$$  


행렬의 곱셈을 이해하는 몇 가지 방식들

- 벡터 $$\times$$ 벡터
- 행렬 $$\times$$ 벡터
- 행렬 $$\times$$ 행렬

### 벡터 $$\times$$ 벡터 (Vector-Vector Products)
#### 내적(inner product or dot product)
두 개의 벡터 $$x, y\in \mathbb{R}^n$$이 주어졌을 때 내적(inner product 또는 dot product) $$x^Ty$$는 다음과 같이 정의된다.  

$$\begin{align*} x^Ty \in \mathbb{R} = [\mbox{ }x_1\mbox{ }x_2\mbox{ }\cdots \mbox{ }x_n\mbox{ }] \begin{bmatrix} y_1\\ y_2\\ \vdots\\ y_n \end{bmatrix} = \sum_{i=1}^n x_i y_i \end{align*}$$  

$$x^Ty = y^Tx$$  

```python
In [11]:import numpy as np
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        x.dot(y)
Out[11]:32

In [12]:y.dot(x)
Out[12]:32
```  

#### 외적(outer product)
두 개의 벡터 $$x\in \mathbb{R}^m, y\in \mathbb{R}^n$$이 주어졌을 때 외적(outer product) $$xy^T\in \mathbb{R}^{m\times n}$$는 다음과 같이 정의된다.  

$$\begin{align*} xy^T \in \mathbb{R}^{m\times n} = \begin{bmatrix} x_1\\ x_2\\ \vdots\\ x_m \end{bmatrix} [\mbox{ }y_1\mbox{ }y_2\mbox{ }\cdots \mbox{ }y_n\mbox{ }] = \begin{bmatrix} x_1y_1 &\ x_1y_2 &\ \cdots &\ x_1y_n\\ x_2y_1 &\ x_2y_2 &\ \cdots &\ x_2y_n\\ \vdots &\ \vdots &\ \ddots &\ \vdots\\ x_my_1 &\ x_my_2 &\ \cdots &\ x_my_n \end{bmatrix} \end{align*}$$  

```python
In [13]:x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

In [14]:x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=0)
        x.shape, y.shape
Out[14]:((3, 1), (1, 3))

In [15]:np.matmul(x,y)
Out[15]:array([[ 4,  5,  6],
                [ 8, 10, 12],
                [12, 15, 18]])
```  

외적이 유용한 경우.  
아래 행렬 $$A$$는 모든 열들이 동일한 벡터 $$x$$를 가지고 있다.  
외적을 이용하면 간편하게 $$x\mathbf{1}^T$$로 나타낼 수 있다 ($$\mathbf{1}\in \mathbb{R}^n$$는 모든 원소가 1인 $$n$$차원 벡터).  

$$\begin{align*} A = \begin{bmatrix} \vert &\ \vert &\ &\ \vert\\ x &\ x &\ \cdots &\ x\\ \vert &\ \vert &\ &\ \vert \end{bmatrix} = \begin{bmatrix} x_1 &\ x_1 &\ \cdots &\ x_1\\ x_2 &\ x_2 &\ \cdots &\ x_2\\ \vdots &\ \vdots &\ \ddots &\ \vdots\\ x_m &\ x_m &\ \cdots &\ x_m \end{bmatrix} = \begin{bmatrix} x_1\\ x_2\\ \vdots\\ x_m \end{bmatrix} \begin{bmatrix} 1 &\ 1 &\ \cdots &\ 1 \end{bmatrix} = x\mathbf{1}^T \end{align*}$$  

```python
In [16]:# column vector
        x = np.expand_dims(np.array([1, 2, 3]), axis=1)

In [17]:ones = np.ones([1,4])

In [18]:A = np.matmul(x, ones)
        A
Out[18]:array([[1., 1., 1., 1.],
                [2., 2., 2., 2.],
                [3., 3., 3., 3.]])
```


### 행렬 $$\times$$ 벡터 (Matrix-Vector Products)

행렬 $$A\in \mathbb{R}^{m\times n}$$와 벡터 $$x\in \mathbb{R}^n$$의 곱은 벡터 $$y = Ax \in \mathbb{R}^m$$이다.  
이 곱을 몇 가지 측면에서 바라볼 수 있다.  
열벡터를 오른쪽에 곱하고($$Ax$$), $$A$$가 행의 형태로 표현되었을 때  

$$\begin{align*} y = Ax = \begin{bmatrix} - & a_1^T & -\\ - & a_2^T & -\\ & \vdots &\\ - & a_m^T & - \end{bmatrix} x = \begin{bmatrix} a_1^Tx\\ a_2^Tx\\ \vdots\\ a_m^Tx \end{bmatrix} \end{align*}$$  

```python
In [19]:A = np.array([
            [1,2,3],
            [4,5,6]
        ])
        A
Out[19]:array([[1, 2, 3],
                [4, 5, 6]])

In [20]:ones = np.ones([3,1])

In [21]:np.matmul(A, ones)
Out[21]:array([[ 6.],
                [15.]])
```  

열벡터를 오른쪽에 곱하고($$Ax$$), $$A$$가 열의 형태로 표현되었을 때  

$$\begin{align*} y = Ax = \begin{bmatrix} \vert &\ \vert &\ &\ \vert\\ a_1 &\ a_2 &\ \cdots &\ a_n\\ \vert &\ \vert &\ &\ \vert \end{bmatrix} \begin{bmatrix} x_1\\ x_2\\ \vdots\\ x_n \end{bmatrix} = \begin{bmatrix} \vert\\ a_1\\ \vert \end{bmatrix} x_1 + \begin{bmatrix} \vert\\ a_2\\ \vert \end{bmatrix} x_2 + \cdots + \begin{bmatrix} \vert\\ a_n\\ \vert \end{bmatrix} x_n \end{align*}$$  

```python
In [22]:A = np.array([
            [1,0,1],
            [0,1,1]
        ])
        x = np.array([
            [1],
            [2],
            [3]
        ])
        np.matmul(A, x)
Out[22]:array([[4],
                [5]])

In [23]:for i in range(A.shape[1]):
            print('a_'+str(i)+':', A[:,i], '\tx_'+str(i)+':', x[i], '\ta_'+str(i)+'*x_'+str(i)+':', A[:,i]*x[i])
Out[23]:a_0: [1 0] 	x_0: [1] 	a_0*x_0: [1 0]
        a_1: [0 1] 	x_1: [2] 	a_1*x_1: [0 2]
        a_2: [1 1] 	x_2: [3] 	a_2*x_2: [3 3]
```  

행벡터를 왼쪽에 곱하고($$x^TA$$), $$A$$가 열의 형태로 표현되었을 때  

$$A\in \mathbb{R}^{m\times n}$$, $$x\in \mathbb{R}^m$$, $$y\in \mathbb{R}^n$$일 때, $$y^T = x^TA$$  

$$\begin{align*} y^T = x^TA = x^T \begin{bmatrix} \vert &\ \vert &\ &\ \vert\\ a_1 &\ a_2 &\ \cdots &\ a_n\\ \vert &\ \vert &\ &\ \vert \end{bmatrix} = \begin{bmatrix} x^Ta_1 &\ x^Ta_2 &\ \cdots &\ x^Ta_n \end{bmatrix} \end{align*}$$  

행벡터를 왼쪽에 곱하고($$x^TA$$), $$A$$가 행의 형태로 표현되었을 때  

$$\begin{align*} y^T =& x^TA\\ =& \begin{bmatrix} x_1 & x_2 & \cdots & x_m \end{bmatrix} \begin{bmatrix} - & a_1^T & -\\ - & a_2^T & -\\ & \vdots &\\ - & a_m^T & - \end{bmatrix}\\ =& x_1 \begin{bmatrix} - & a_1^T & - \end{bmatrix} + x_2 \begin{bmatrix} - & a_2^T & - \end{bmatrix} + \cdots + x_n \begin{bmatrix} - & a_n^T & - \end{bmatrix} \end{align*}$$  

# 행렬 $$\times$$ 행렬 (Matrix-Matrix Products)
강의 후, 여기서 부터 시작





![lambda](/assets/images/decision-region.png){: .align-center .img-50}  


# Appendix
## MathJax

$${\sim}A$$  
$$\sim{A}$$:  
```
$${\sim}A$$
$$\sim A$$
```  
align  
$$\begin{align*} A = \begin{bmatrix} a_{11} &\ a_{12} &\ \cdots &\ a_{1n}\\ a_{21} &\ a_{22} &\ \cdots &\ a_{2n}\\ \vdots &\ \vdots &\ \ddots &\ \vdots\\ a_{m1} &\ a_{m2} &\ \cdots &\ a_{mn} \end{bmatrix} \end{align*}$$  

$$\begin{align*} A = \begin{bmatrix} \vert &\ \vert &\ &\ \vert\\ a_1 &\ a_2 &\ \cdots &\ a_n\\ \vert &\ \vert &\ &\ \vert \end{bmatrix} \end{align*}$$  

$$\begin{align*} A = \begin{bmatrix} - & a_1^T & -\\ - & a_2^T & -\\ & \vdots &\\ - & a_m^T & - \end{bmatrix} \end{align*}$$:  
```
$$\begin{align*} A = \begin{bmatrix} a_{11} &\ a_{12} &\ \cdots &\ a_{1n}\\ a_{21} &\ a_{22} &\ \cdots &\ a_{2n}\\ \vdots &\ \vdots &\ \ddots &\ \vdots\\ a_{m1} &\ a_{m2} &\ \cdots &\ a_{mn} \end{bmatrix} \end{align*}$$  

$$\begin{align*} A = \begin{bmatrix} \vert &\ \vert &\ &\ \vert\\ a_1 &\ a_2 &\ \cdots &\ a_n\\ \vert &\ \vert &\ &\ \vert \end{bmatrix} \end{align*}$$  

$$\begin{align*} A = \begin{bmatrix} - & a_1^T & -\\ - & a_2^T & -\\ & \vdots &\\ - & a_m^T & - \end{bmatrix} \end{align*}$$  
```  




가변 괄호 with escape curly brackets    
$$\left\{-\frac{1}{2\sigma^{2}} \sum_{n=1}^{N}(x_{n}-\mu)^{2} \right\}$$:  
```
$$\left\{-\frac{1}{2\sigma^{2}} \sum_{n=1}^{N}(x_{n}-\mu)^{2} \right\}$$ 
```  


## References

> Pattern Recognition and Machine Learning: <https://tensorflowkorea.files.wordpress.com/2018/11/bishop-pattern-recognition-and-machine-learning-2006.pdf>  

