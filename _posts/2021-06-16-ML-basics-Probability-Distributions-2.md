---
date: 2021-06-20 00:35
title: "ML basics - Probability Distributions 2"
categories: DevCourse2 ProbabilityDistributions MathJax
tags: DevCourse2 ProbabilityDistributions MathJax
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---


# 가우시안 분포 (Gaussian Distribution)
![gaussian-dist](/assets/images/gaussian-dist.png){: .align-center}  

a single variable $$x$$:  

$$\mathcal{N}(x \vert \mu, \sigma^2) = \frac{1}{(2\pi \sigma^2)^{1/2}}\exp \left\{ -\frac{1}{2\sigma^2}(x-\mu)^2 \right\}$$  

a D-dimensional vector $$\boldsymbol{x}$$:  

$$\mathcal{N}(\boldsymbol{x} \vert \boldsymbol{\mu}, \Sigma) = \frac{1}{(2\pi)^{D/2}}\frac{1}{\vert \Sigma \vert^{1/2}}\exp \left\{ -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) \right\}$$  

- $$\boldsymbol{x}$$ is a $$D$$-dimensional mean vector.
- $$\Sigma$$ is a $$D \times D$$ covariance matrix.
- $$\vert \Sigma \vert$$ denotes the determinant of $$\Sigma$$.

$$\frac{1}{(2\pi)^{D/2}}\frac{1}{\vert \Sigma \vert^{1/2}}\exp \left\{ -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) \right\}$$  

이 값은 scalar 값이라는 사실을 볼 수 있다.  
이유:  
- $$\Sigma$$가 행렬이지만 여기서 determinant($$\vert \Sigma \vert$$)을 구했기 때문에.
- 지수부($$\exp$$)는 이차형식이기 때문에 역시 scalar.  
따라서 결과 값은 scalar값이다.

> 단일변수 $$x$$ 이던지 $$D$$차원의 벡터이던지 간에 가우시안 분포가 주어졌을 때 어떤 것이 확률변수이고 확률의 파라미터인지 잘 구분하는 것이 중요하다. $$x$$가 확률변수이고 $$\mu$$와 $$\Sigma$$가 파라미터. 유의할 점은 가우시간 분포가 주어졌을 때, 평균과 분산이 주어진 것이 아니고, 이것들이 파라미터로 주어진 확률밀도 함수의 평균과 공분산이 $$\mu$$와 $$\Sigma(sum)$$이 된다는 것이다.


유의할 점은 가우시간 분포가 주어졌을 때, 평균과 분산이 주어진 것이 아니고,
이것들이 파라미터로 주어진 확률밀도 함수의 평균과 공분산이 $$\boldsymbol{\mu}$$와 $$\Sigma$$이 된다는 것이다.


## 가우시안 분포가 일어나는 여러가지 상황
### 1. 정보이론에서 엔트로피를 최대화시키는 확률분포
{% include video id="2s3aJfRr9gE" provider="youtube" %}  
영상보고 정리할 것
### 2. 중심극한정리
{% include video id="JNm3M9cqWyc" provider="youtube" %}  
영상보고 정리할 것


## 가우시안 분포의 기하학적인 형태
- $$\boldsymbol{x}$$에 대한 함수적 종속성은 지수부에 등장하는 이차형식(quadratic form)에 있다.  

$$\Delta^2 = (\boldsymbol{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})$$  

> $$(\boldsymbol{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})$$ 이러한 형태의 이차형식이 가우시안 분포에서 핵심적인 부분이다.  

- $$\Sigma$$가 공분산으로 주어진 것이 아니기 때문에 처음부터 이 행렬이 대칭이라고 생각할 필요는 없다. 하지만 이차형식에 나타나는 행렬은 **오직 대칭부분**만이 그 값에 기여한다는 사실을 기억하자.  

$$x^TAx = (x^TAx)^T = x^TA^Tx = x^T\left( \frac{1}{2}A + \frac{1}{2}A^T \right)x$$  

따라서 $$\Sigma$$가 대칭행렬인 것으로 간주할 수 있다.  

- **대칭행렬**의 성질에 따라서 $$\Sigma$$를 아래와 같이 나타낼 수 있다.  

$$\Sigma = U^T\Lambda U$$  

$$U = \begin{bmatrix} - \boldsymbol u_{1}^{T} - \cr - \boldsymbol u_{2}^{T} - \cr \vdots \cr - \boldsymbol u_{D}^{T} - \end{bmatrix}, \Lambda = diag(\lambda_{1},\cdots,\lambda_{D})$$  

$$\begin{align} \Sigma &= U^T\Lambda U \\\\ &= \begin{bmatrix} \vert &\ \vert &\ &\ \vert\\ \boldsymbol u_{1} &\ \boldsymbol u_{2} &\ \cdots &\ \boldsymbol u_{D}\\ \vert &\ \vert &\ &\ \vert \end{bmatrix} diag(\lambda_{1},\cdots,\lambda_{D}) \begin{bmatrix} - \boldsymbol u_{1}^{T} - \cr - \boldsymbol u_{2}^{T} - \cr \vdots \cr - \boldsymbol u_{D}^{T} - \end{bmatrix} \\\\ &= \begin{bmatrix} \vert &\ \vert &\ &\ \vert\\ \boldsymbol u_{1}\lambda_{1} &\ \boldsymbol u_{2}\lambda_{2} &\ \cdots &\ \boldsymbol u_{D}\lambda_{D}\\ \vert &\ \vert &\ &\ \vert \end{bmatrix} \begin{bmatrix} - \boldsymbol u_{1}^{T} - \cr - \boldsymbol u_{2}^{T} - \cr \vdots \cr - \boldsymbol u_{D}^{T} - \end{bmatrix} \\\\ &= \sum_{i=1}^{D}\lambda_{i}\boldsymbol u_{i}\boldsymbol u_{i}^{T} \end{align}$$  

- $$\Sigma^{-1}$$도 쉽게 구할 수 있다.  

$$\Sigma^{-1} = \sum_{i=1}^{D}\frac{1}{\lambda_{i}}\boldsymbol u_{i}\boldsymbol u_{i}^{T}$$  

- 이차형식은 다음과 같이 표현될 수 있다.  

$$\Delta^2 = \sum_{i=1}^{D}\frac{y_{i}^2}{\lambda_{i}}$$  

$$y_{i} = \boldsymbol u_{i}^{T}(\boldsymbol x - \boldsymbol \mu)$$  

![gaussian-quadratic-form](/assets/images/gaussian-quadratic-form.png){: .align-center .img-80}  

- 벡터식으로 확장하면

$$\boldsymbol y = U(\boldsymbol x - \boldsymbol \mu)$$  

![gaussian-quadratic-form-vector](/assets/images/gaussian-quadratic-form-vector.png){: .align-center .img-80}  

- $$\boldsymbol y$$를 벡터들 $$\boldsymbol \mu_{i}$$에 의해 정의된 새로운 좌표체계 내의 점으로 해석할 수 있다. 이것을 기저변환(change of basis)이라고 한다.  

$$\begin{align} \boldsymbol y &= U(\boldsymbol x - \boldsymbol \mu) \\ \boldsymbol x - \boldsymbol \mu &= U^{-1}\boldsymbol y \\ &= U^{T}\boldsymbol y \\ &= \begin{bmatrix} \vert &\ \vert &\ &\ \vert\\ \boldsymbol u_{1} &\ \boldsymbol u_{2} &\ \cdots &\ \boldsymbol u_{D}\\ \vert &\ \vert &\ &\ \vert \end{bmatrix}\boldsymbol y \end{align}$$  

  - $$\boldsymbol x - \boldsymbol \mu$$: standard basis에서의 좌표
  - $$\boldsymbol y$$: basis $$\{\boldsymbol u_{1}, \boldsymbol u_{2}, \cdots, \boldsymbol u_{D} \}$$에서의 좌표  

![elliptical surface of constant proba-bility density for a Gaussian ina two-dimensional space](/assets/images/elliptical surface of constant proba-bility density for a Gaussian ina two-dimensional space.png){: .align-center}  

$$\boldsymbol{y}$$는 coefficient들을 모아논 값.  
red circle: x1, x2 좌표내에서 점들의 집합이 **동일한** 분포를 갖는다.  
고유벡터 $$u_1 \sim u_d$$까지를 basis로 하는 좌표계에서는 차원을 이루고
그 모양은 고유값들에 의해 결정된다.  

## 가우시안 분포의 Normalization 증명
- 확률이론시간에 배운 확률변수의 함수를 복습하라. $$\boldsymbol y$$의 확률밀도함수를 구하기 위해서 Jacobian $$\boldsymbol J$$를 구해야 한다.  

$$J_{ij} = \frac{\partial x_{i}}{\partial y_{i}} = U_{ij} = (U^{T})_{ij}$$  

![jacobian-x-y](/assets/images/jacobian-x-y.png){: .align-center}  

> 결국 $$\boldsymbol J = U^T$$라는 사실을 알 수 있다.  

$$\vert \boldsymbol J \vert ^2 = \vert U^T\vert ^2 = \vert U^T\vert\vert U\vert = \vert U^{T}U\vert = \vert \boldsymbol I\vert = 1$$  

- 행렬식 $$\vert \Sigma\vert$$는 고유값의 곱으로 나타낼 수 있다.  

$$\vert \Sigma\vert^{1/2} = \prod_{j=1}^{D} \lambda_{i}^{1/2}$$  

- 따라서, $$\boldsymbol y$$의 확률밀도함수는  

$$p(\boldsymbol y) = p(\boldsymbol x)\vert \boldsymbol J\vert = \prod_{j=1}^{D}\frac{1}{(2\pi\lambda_{j})^{1/2}}\exp\left\{ -\frac{y_{j}^{2}}{2\lambda_{j}} \right\}$$  

![gaussian-y-pdf](/assets/images/gaussian-y-pdf.png){: .align-center .img-80}  

- $$\boldsymbol y$$의 normalization  

$$\int p(\boldsymbol y)d\boldsymbol y = \prod_{j=1}^{D}\int_{-\infty}^{\infty}\frac{1}{(2\pi\lambda_{j})^{1/2}}\exp\left\{ -\frac{y_{j}^{2}}{2\lambda_{j}} \right\}dy_{j}=1$$  

> 적분을 하게 되면 D개의 y variables이 각각 다른 변수들이기 때문에 D개의 적분의 곱으로 나타낼 수 있다.  
> 따라서단일변수의 가우시안 분포와 같다. 이미 단일 변수일 경우 normalize되었다는 것을 확인했기 때문에 이를 D번 곱해주는 결과와 같게 된다.  

## 가우시안 분포의 기댓값
### 다변량(multivariate) 확률변수의 기댓값
- \\(\boldsymbol x = (x_{1},\cdots,x_{n})^T\\)
- \\(\mathbb{E}[\boldsymbol x] = (\mathbb{E}[x_{1}],\cdots,\mathbb{E}[x_{n}])^T\\)
- \\(\mathbb{E}[x_{1}] = \int x_{1}p(x_1)dx_1 = \int x_1 (\int p(x_1, \cdots, x_n)dx_2, \cdots, dx_n)dx_1 = \int x_1 p(x_1,\cdots,x_n)dx_1, \cdots, dx_n\\)

$$\begin{align}  \mathbb{E}[\boldsymbol x] &= \frac{1}{(2\pi)^{D/2}}\frac{1}{\vert \Sigma \vert^{1/2}}\int \exp \left\{ -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) \right\}\boldsymbol x d \boldsymbol x \\\\ &= \frac{1}{(2\pi)^{D/2}}\frac{1}{\vert \Sigma \vert^{1/2}}\int \exp \left\{ -\frac{1}{2}\boldsymbol z^T\Sigma^{-1}\boldsymbol z \right\}(\boldsymbol z + \boldsymbol \mu)d\boldsymbol z  &\ \mathrm{by}~ \boldsymbol z = \boldsymbol{x}-\boldsymbol{\mu} \end{align}$$  

$$\begin{align} \int \exp \left\{ -\frac{1}{2}\boldsymbol z^T\Sigma^{-1}\boldsymbol z \right\}\boldsymbol{z}d\boldsymbol{z} &= \int\vert \boldsymbol J\vert \exp \left\{ -\frac{1}{2} \sum_{i=1}^{D} \frac{y_{i}^{2}}{\lambda_{i}} \right\}\boldsymbol{y}d\boldsymbol{y} &\ \mathrm{by}~ \boldsymbol z = \sum_{j=1}^{D}y_{j}\boldsymbol{u}_{j}, y_{j} = \boldsymbol{U}_{j}^{T}\boldsymbol{z}\\\\ &= \begin{bmatrix} \int \exp\left\{ -\frac{1}{2}\sum_{i=1}^{D}\frac{y_{i}^{2}}{\lambda_{i}}y_{1}dy_{1},\cdots,dy_{D} \right\} \cr \int \exp\left\{ -\frac{1}{2}\sum_{i=1}^{D}\frac{y_{i}^{2}}{\lambda_{i}}y_{D}dy_{1},\cdots,dy_{D} \right\} \end{bmatrix} \end{align}$$ 

$$\begin{align} \int \exp\left\{ -\frac{1}{2}\sum_{i=1}^{D}\frac{y_{i}^{2}}{\lambda_{i}}\right\}y_{1}dy_{1},\cdots,dy_{D} &= \int\left( \int_{0}^{\infty}\exp\left\{ -\frac{1}{2}\sum_{i=1}^{D}\frac{y_{i}^{2}}{\lambda_{i}}\right\}y_{1}dy_{1} + \int_{-\infty}^{0}\exp\left\{ -\frac{1}{2}\sum_{i=1}^{D}\frac{y_{i}^{2}}{\lambda_{i}}\right\}y_{1}dy_{1} \right)dy_{2}\cdots dy_{D} \\\\ &= \int\left( \int_{0}^{\infty}\exp\left\{ -\frac{1}{2}\sum_{i=1}^{D}\frac{y_{i}^{2}}{\lambda_{i}}\right\}y_{1}dy_{1} + \int_{0}^{\infty}\exp\left\{ -\frac{1}{2}\sum_{i=1}^{D}\frac{y_{i}^{2}}{\lambda_{i}}\right\}(-y_{1})dy_{1} \right)dy_{2}\cdots dy_{D} \\\\ &= 0  \end{align}$$  

따라서  

$$\mathbb{E}[\boldsymbol{x}] = \boldsymbol \mu$$  

## 가우시안 분포의 공분산
공분산을 구하기 위해서 먼저 2차 적률(second order moments)을 구한다.  

$$\begin{align}  \mathbb{E}[\boldsymbol{x}] &= \frac{1}{(2\pi)^{D/2}}\frac{1}{\vert \Sigma \vert^{1/2}}\int \exp \left\{ -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) \right\}\boldsymbol{x}\boldsymbol{x}^{T} d \boldsymbol{x} \\\\ &= \frac{1}{(2\pi)^{D/2}}\frac{1}{\vert \Sigma \vert^{1/2}}\int \exp \left\{ -\frac{1}{2}\boldsymbol{z}^T\Sigma^{-1}\boldsymbol{z} \right\}(\boldsymbol{z} + \boldsymbol{\mu})(\boldsymbol{z} + \boldsymbol{\mu})^{T}d\boldsymbol{z}   \end{align}$$  

$$\mathbb{E}[\boldsymbol x\boldsymbol x^{T}]$$는 $$D \times D$$ 행렬임을 기억하라. $$\boldsymbol z$$를 $$U^{T}\boldsymbol y$$로 치환하면  

$$\frac{1}{(2\pi)^{D/2}}\frac{1}{\vert \Sigma \vert^{1/2}}\int \exp \left\{ -\frac{1}{2}\boldsymbol{z}^T\Sigma^{-1}\boldsymbol{z} \right\}\boldsymbol{z}\boldsymbol{z}^{T}d\boldsymbol{z} = \frac{1}{(2\pi)^{D/2}}\frac{1}{\vert \Sigma \vert^{1/2}}\sum_{i=1}^{D}\sum_{j=1}^{D}\boldsymbol{u}_{i}\boldsymbol{u}_{j}^{T} \int \exp \left\{ -\frac{1}{2} \sum_{k=1}^{D} \frac{y_{k}^{2}}{\lambda_{k}} \right\}y_{i}y_{j}d\boldsymbol{y}$$  

행렬 $$\frac{1}{(2\pi)^{D/2}}\frac{1}{\vert \Sigma \vert^{1/2}}\sum_{i=1}^{D}\sum_{j=1}^{D}\boldsymbol{u}_{i}\boldsymbol{u}_{j}^{T} \int \exp \left\{ -\frac{1}{2} \sum_{k=1}^{D} \frac{y_{k}^{2}}{\lambda_{k}} \right\}y_{i}y_{j}d\boldsymbol{y}$$은 $$D^2$$개의 행렬의 합인데 그 중 $$i \neq j$$인 모든 경우 영행렬이 된다.  

$$\int^{i \neq j} \cdots \int \exp \left\{ -\frac{1}{2} \sum_{k=1}^{D} \frac{y_{k}^{2}}{\lambda_{k}} \right\}y_{i}y_{j}d\boldsymbol{y} = \int \cdots y_{i}\left[ \int \exp \left\{ -\frac{1}{2} \sum_{k=1}^{D} \frac{y_{k}^{2}}{\lambda_{k}} \right\}y_{i}dy_{i} \right]d\boldsymbol{y}\setminus \{y_{i}\}$$  
$$= 0$$  

따라서,  

$$\begin{align}   \frac{1}{(2\pi)^{D/2}}\frac{1}{\vert \Sigma \vert^{1/2}}\sum_{i=1}^{D}\sum_{j=1}^{D}\boldsymbol{u}_{i}\boldsymbol{u}_{j}^{T} \int \exp \left\{ -\frac{1}{2} \sum_{k=1}^{D} \frac{y_{k}^{2}}{\lambda_{k}} \right\}y_{i}y_{j}d\boldsymbol{y} &= \frac{1}{(2\pi)^{D/2}}\frac{1}{\vert \Sigma \vert^{1/2}}\sum_{i=1}^{D}\sum_{i=1}^{D}\boldsymbol{u}_{i}\boldsymbol{u}_{i}^{T} \int \exp \left\{ -\frac{1}{2} \sum_{k=1}^{D} \frac{y_{k}^{2}}{\lambda_{k}} \right\}y_{i}^{2}d\boldsymbol{y} \\ &= \sum_{i=1}^{D}\boldsymbol{u}_{i}\boldsymbol{u}_{i}^{T} \left[ \frac{1}{(2\pi)^{D/2}}\prod_{j=1}^{D}\frac{1}{\lambda_{j}^{1/2}} \right]\left[ \prod_{k=\{1,\cdots,D\}\setminus\{i\}} \int \exp \left\{ -\frac{1}{2}\frac{y_{k}^{2}}{\lambda_{k}} \right\}dy_{k}  \right]\left[ \int\exp\left\{ -\frac{1}{2}\frac{y_{k}^{2}}{\lambda_{k}} \right\}y_{i}^{2}dy_{i} \right] \\\\ &= \sum_{i=1}^{D}\boldsymbol{u}_{i}\boldsymbol{u}_{i}^{T}\lambda_{i}\\\\ &= \Sigma  \end{align}$$

> $${k=\{1,\cdots,D\}\setminus\{i\}}$$: set - set의 의미이다.  

$$\mathbb{E}[\boldsymbol{x}\boldsymbol{x}^{T}] = \boldsymbol{\mu}\boldsymbol{\mu}^{T} + \Sigma$$  

확률변수의 벡터 $$\boldsymbol{x}$$를 위한 공분산은 다음과 같이 정의된다.  

$$cov[\boldsymbol{x}] = \mathbb{E}[(\boldsymbol{x}-\mathbb{E}[\boldsymbol{x}])(\boldsymbol{x}-\mathbb{E}[\boldsymbol{x}])^{T}]$$  

위의 결과를 이용하면 공분산은  

$$cov[\boldsymbol{x}] = \Sigma$$  


## 조건부 가우시안 분포 (Conditional Gaussian Distribution)
$$D$$차원의 확률변수 벡터 $$\boldsymbol{x}$$가 가우시안 분포 $$\mathcal{N}(\boldsymbol{x} \vert \boldsymbol{\mu}, \sigma)$$를 따른다고 하자. $$\boldsymbol{x}$$를 두 그룹의 확률변수들로 나누었을 때, 한 그룹이 주어졌을 때 나머지 그룹의 조건부 확률도 가우시안 분포를 따르고, 각 그룹의 주변확률 또한 가우시안 분포를 따른다는 것을 보이고자 한다.  

$$\boldsymbol{x}$$가 다음과 같은 형태를 가진다고 하자.  

$$\boldsymbol{x} = \begin{bmatrix} \boldsymbol{x}_{a} \cr \boldsymbol{x}_{b} \end{bmatrix}$$  

$$\boldsymbol{x}_{a}$$는 $$M$$개의 원소를 가진다고 하자. 그리고 평균 벡터와 공분산 행렬은 다음과 같이 주어진다고 하자.  

$$\boldsymbol{\mu} = \begin{bmatrix} \boldsymbol{\mu}_{a} \cr \boldsymbol{\mu}_{b} \end{bmatrix}$$  

$$\Sigma = \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab}  \cr \Sigma_{ba} & \Sigma_{bb} \end{bmatrix}$$  

때로는 공분산의 역행렬, 즉 정확도 행렬(precision matrix)을 사용하는 것이 수식을 간편하게 한다.  

$$\Lambda = \Sigma^{-1}$$  

$$\Lambda = \begin{bmatrix} \Lambda{aa} & \Lambda{ab}  \cr \Lambda{ba} & \Lambda{bb} \end{bmatrix}$$

지수부의 이차형식을 위의 파티션을 사용해서 전개해보자.  

$$-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})$$  

$$= -\frac{1}{2}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})^{T}\Lambda_{aa}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})-\frac{1}{2}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})^{T}\Lambda_{ab}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})\\\\ -\frac{1}{2}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})^{T}\Lambda_{ba}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})-\frac{1}{2}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})^{T}\Lambda_{bb}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})$$  

### 완전제곱식 (Completing the Square) 방법
확률밀도함수 $$p(\boldsymbol{x}_{a}, \boldsymbol{x}_{b})$$를 $$p(\boldsymbol{x}_{a}, \boldsymbol{x}_{b}) = g(\boldsymbol{x}_{a}\alpha)$$로 나타낼 수 있다고 하자. 여기서 $$\alpha$$는 $$\boldsymbol{x}_{a}$$와 독립적이고 $$\int g(\boldsymbol{x}_{a})d\boldsymbol{x}_{a}=1$$이다. 따라서  

$$\begin{align}   \int p(\boldsymbol{x}_{a}, \boldsymbol{x}_{b})d\boldsymbol{x}_{a} &= \int g(\boldsymbol{x}_{a})\alpha d\boldsymbol{x}_{a} \\ &= \alpha \int g(\boldsymbol{x}_{a}) d\boldsymbol{x}_{a} \\ &= \alpha \end{align}$$  

$$\alpha = p(\boldsymbol{x}_{b})$$  
$$p(\boldsymbol{x}_{a}, \boldsymbol{x}_{b})$$  
$$g(\boldsymbol{x}_{a}) = p(\boldsymbol{x}_{a}\vert \boldsymbol{x}_{b})$$  

위 과정을 통해 함수 $$g(\boldsymbol{x}_{a})$$를 찾는 것이 목표이다.  

가우시안 분포의 지수부는 다음과 같이 전개된다는 것이 중요한 포인트이다.  

$$\begin{align} -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) &= -\frac{1}{2}(\boldsymbol{x}^{T}-\boldsymbol{\mu}^{T})\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\\ &= -\frac{1}{2}(\boldsymbol{x}^{T}\Sigma^{-1}-\boldsymbol{\mu}^{T}\Sigma^{-1})(\boldsymbol{x}-\boldsymbol{\mu})\\ &= -\frac{1}{2}(\boldsymbol{x}^{T}\Sigma^{-1}\boldsymbol{x}-\boldsymbol{\mu}^{T}\Sigma^{-1}\boldsymbol{x}-\boldsymbol{x}^{T}\Sigma^{-1}\boldsymbol{\mu}+\boldsymbol{\mu}^{T}\Sigma^{-1}\boldsymbol{\mu})\\ &= -\frac{1}{2}\boldsymbol{x}^{T}\Sigma^{-1}\boldsymbol{x}+\boldsymbol{x}^{T}\Sigma^{-1}\boldsymbol{\mu} + const   \end{align}$$  

여기서 상수부 $$const$$는 $$\boldsymbol{x}$$와 독립된 항들을 모은 것이다. 따라서 어떤 복잡한 함수라도 지수부를 정리했을 때 $$-\frac{1}{2}\boldsymbol{x}^{T}\Sigma^{-1}\boldsymbol{x}+\boldsymbol{x}^{T}\Sigma^{-1}\boldsymbol{\mu} + const$$의 형태가 된다면 이 함수는 공분산 행렬 $$\Sigma$$와 평균벡터 $$\boldsymbol{\mu}$$를 가지는 가우시안 분포임을 알 수 있다.  $$\boldsymbol{x}$$에 관한 이차항과 일차항의 계수를 살피면 된다는 것이다.  

$$-\frac{1}{2}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})^{T}\Lambda_{aa}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})-\frac{1}{2}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})^{T}\Lambda_{ab}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})-\frac{1}{2}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})^{T}\Lambda_{ba}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})-\frac{1}{2}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})^{T}\Lambda_{bb}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})$$에서 $$\boldsymbol{x}_{a}$$의 이차항은  

$$-\frac{1}{2}\boldsymbol{x}_{a}^{T}\Lambda_{aa}\boldsymbol{x}_{a}$$  

이다. 따라서 공분산은  

$$\sum_{a\vert b} = \Lambda_{aa}^{-1}$$  

이다. 이제 평균벡터를 구하기 위해서는 $$\boldsymbol{x}_{a}$$의 일차항을 정리하면 된다. $$\boldsymbol{x}_{a}$$의 일차항은  

$$\boldsymbol{x}_{a}^{T}\{ \Lambda_{aa}\boldsymbol{\mu}_{a} - \Lambda_{ab}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})\}$$  

$$\boldsymbol{x}_{a}$$의 일차항의 계수는 $$\Sigma_{a\vert b}^{-1}\boldsymbol{\mu}_{a\vert b}$$ 이어야 하므로  

$$\begin{align} \boldsymbol{\mu}_{a\vert b} &= \Sigma_{a\vert b} \{ \Lambda_{aa}\boldsymbol{\mu}_{a} - \Lambda_{ab}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})\}\\ &= \boldsymbol{\mu}_{a} - \Lambda_{aa}^{-1}\Lambda_{ab}(\boldsymbol{x}_{b} - \boldsymbol{\mu}_{b}) \end{align}$$

## 주변 가우시안 분포 (Marginal Gaussian Distribution)
다음과 같은 주변분포를 계산하고자 한다.  

$$p(\boldsymbol{x}_{a}) = \int p(\boldsymbol{x}_{a}, \boldsymbol{x}_{b})d\boldsymbol{x}_{b}$$  

전략은 다음과 같다.  

$$\begin{align} p(\boldsymbol{x}_{a}) &= \int p(\boldsymbol{x}_{a}, \boldsymbol{x}_{b})d\boldsymbol{x}_{b} \\ &= \int\alpha\exp\left\{ -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) \right\} d\boldsymbol{x}_{b} \\ &= \int\alpha\exp\{ f(\boldsymbol{x}_{b},\boldsymbol{x}_{a})+g(\boldsymbol{x}_{a})+const \}d\boldsymbol{x}_{b} \\ &= \int\alpha\exp\{ f(\boldsymbol{x}_{b},\boldsymbol{x}_{a})-\tau+\tau+g(\boldsymbol{x}_{a})+const \}d\boldsymbol{x}_{b}\\ &= \int\alpha\exp\{ f(\boldsymbol{x}_{b},\boldsymbol{x}_{a})-\tau\}\exp\{\tau+g(\boldsymbol{x}_{a})+const \}d\boldsymbol{x}_{b} \\ &= \alpha\exp\{\tau+g(\boldsymbol{x}_{a})+const \}\int\exp\{f(\boldsymbol{x}_{b},\boldsymbol{x}_{a})-\tau\}d\boldsymbol{x}_{b} \\ &= \alpha\beta\exp\{\tau+g(\boldsymbol{x}_{a})+const \} \end{align}$$


- 위에서 함수 $$f(\boldsymbol{x}_{b},\boldsymbol{x}_{a})$$는 원래 지수부를 $$\boldsymbol{x}_{a},\boldsymbol{x}_{b}$$파티션을 통해 전개한 식 중에서 $$\boldsymbol{x}_{b}$$을 포함한 모든 항들을 모은 식이다. 그리고 $$g(\boldsymbol{x}_{a})$$는 $$f(\boldsymbol{x}_{b},\boldsymbol{x}_{a})$$에 포함된 항들을 제외한 항들 중 $$\boldsymbol{x}_{a}$$를 포함한 모든 항들을 모은식이다. $$const$$는 나머지 항들을 모은 식이다.
- $$f(\boldsymbol{x}_{b},\boldsymbol{x}_{a})-\tau$$는 $$\boldsymbol{x}_{b}$$을 위한 완전제곱식이다.
- $$\alpha\exp\{\tau+g(\boldsymbol{x}_{a})+const\}$$는 $$\boldsymbol{x}_{b}$$와 독립적이므로 적분식 밖으로 나갈 수 있다.
- $$\tau+g(\boldsymbol{x}_{a})$$를 $$\boldsymbol{x}_{a}$$의 완전제곱식으로 만들면 $$\boldsymbol{x}_{b}$$의 평균벡터와 공분산행렬을 구할 수 있다.

파티션을 위한 이차형식을 다시 살펴본다.  

$$-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})$$  

$$= -\frac{1}{2}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})^{T}\Lambda_{aa}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})-\frac{1}{2}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})^{T}\Lambda_{ab}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})\\\\ -\frac{1}{2}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})^{T}\Lambda_{ba}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})-\frac{1}{2}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})^{T}\Lambda_{bb}(\boldsymbol{x}_{b}-\boldsymbol{\mu}_{b})$$    

이 식을 전부 펼치면 총 16개의 항이 있게 된다. 이 중 $$\boldsymbol{x}_{b}$$를 포함한 항들의 개수는 7개, 나머지 중에서 $$\boldsymbol{x}_{ㅁ}$$를 가진 항의 개수는 5개이다.  

$$\begin{align}f(\boldsymbol{x}_{b},\boldsymbol{x}_{a}) &= -\frac{1}{2}\boldsymbol{x}_{b}^{T}\Lambda_{bb}\boldsymbol{x}_{b}+\frac{1}{2}\boldsymbol{x}_{b}^{T}\Lambda_{bb}\boldsymbol{\mu}_{b}+\frac{1}{2}\boldsymbol{\mu}_{b}^{T}\Lambda_{bb}\boldsymbol{x}_{b}-\frac{1}{2}\boldsymbol{x}_{b}^{T}\Lambda_{ba}\boldsymbol{x}_{a}+\frac{1}{2}\boldsymbol{x}_{b}^{T}\Lambda_{ba}\boldsymbol{\mu}_{a}-\frac{1}{2}\boldsymbol{x}_{a}^{T}\Lambda_{ab}\boldsymbol{x}_{b}+\frac{1}{2}\boldsymbol{\mu}_{a}^{T}\Lambda_{ab}\boldsymbol{x}_{b} \\ &= -\frac{1}{2}\boldsymbol{x}_{b}^{T}\Lambda_{bb}\boldsymbol{x}_{b} + \boldsymbol{x}_{b}^{T}\{ \Lambda_{bb}\boldsymbol{\mu}_{b} - \Lambda_{ba}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})\} \end{align}$$  

$$\begin{align}g(\boldsymbol{x}_{a}) &= \frac{1}{2}\boldsymbol{\mu}_{b}^{T}\Lambda_{ba}\boldsymbol{x}_{a}+\frac{1}{2}\boldsymbol{x}_{a}^{T}\Lambda_{ab}\boldsymbol{\mu}_{b}-\frac{1}{2}\boldsymbol{x}_{a}^{T}\Lambda_{aa}\boldsymbol{x}_{a}+\frac{1}{2}\boldsymbol{x}_{a}^{T}\Lambda_{aa}\boldsymbol{\mu}_{a}+\frac{1}{2}\boldsymbol{\mu}_{a}^{T}\Lambda_{aa}\boldsymbol{x}_{a} \\ &= -\frac{1}{2}\boldsymbol{x}_{a}^{T}\Lambda_{aa}\boldsymbol{x}_{a} + \boldsymbol{x}_{a}^{T} (\Lambda_{aa}\boldsymbol{\mu}_{a}+\Lambda_{ab}\boldsymbol{\mu}_{b})\end{align}$$  


아래와 같이 $$f(\boldsymbol{x}_{b},\boldsymbol{x}_{a})$$를 완전제곱식으로 만든다.  

$$-\frac{1}{2}\boldsymbol{x}_{b}^{T}\Lambda_{bb}\boldsymbol{x}_{b}+\boldsymbol{x}_{b}^{T}\boldsymbol{m} = -\frac{1}{2}(\boldsymbol{x}_{b}-\Lambda_{bb}^{-1}\boldsymbol{m})^{T}\Lambda_{bb}(\boldsymbol{x}_{b}-\Lambda_{bb}^{-1}\boldsymbol{m})+\frac{1}{2}\boldsymbol{m}^{T}\Lambda_{bb}^{-1}\boldsymbol{m}$$  

where $$\boldsymbol{m} = \Lambda_{bb}\boldsymbol{\mu}_{b} - \Lambda_{ba}(\boldsymbol{x}_{a}-\boldsymbol{\mu}_{a})$$  

따라서 $$\tau = \frac{1}{2}\boldsymbol{m}^{T}\Lambda_{bb}^{-1}\boldsymbol{m}$$이고  

$$\int\exp\{f(\boldsymbol{x}_{b},\boldsymbol{x}_{a})-\tau\}d\boldsymbol{x}_{b} = \int\exp\left\{ -\frac{1}{2}(\boldsymbol{x}_{b}-\Lambda_{bb}^{-1}\boldsymbol{m})^{T}\Lambda_{bb}(\boldsymbol{x}_{b}-\Lambda_{bb}^{-1}\boldsymbol{m}) \right\}d\boldsymbol{x}_{b}$$  

이 값은 공분산 $$\Lambda_{bb}$$에만 종속되고 $$\boldsymbol{x}_{a}$$에 독립적이므로 $$\alpha\beta\exp\{\tau+g(\boldsymbol{x}_{a})+const\}$$의 지수부에만 집중하면 된다.  

마지막으로 $$\tau+g(\boldsymbol{x}_{a})+const$$를 살펴보자.  

$$\begin{align}\tau+g(\boldsymbol{x}_{a})+const &= \frac{1}{2}\boldsymbol{m}^{T}\Lambda_{bb}^{-1}\boldsymbol{m} - \frac{1}{2}\boldsymbol{x}_{a}^{T}\Lambda_{aa}\boldsymbol{x}_{a} + \boldsymbol{x}_{a}^{T}(\Lambda_{aa}\boldsymbol{\mu}_{a} + \Lambda_{ab}\boldsymbol{\mu}_{b})+const \\ &= -\frac{1}{2}\boldsymbol{x}_{a}^{T}(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\boldsymbol{x}_{a} + \boldsymbol{x}_{a}^{T}(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\boldsymbol{\mu}_{a}+const  \end{align}$$  

따라서 공분산은  

$$\Sigma_{a} = (\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})^{-1}$$  

이고, 평균 벡터는  

$$\Sigma_{a}(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\boldsymbol{\mu}_{a} = \boldsymbol{\mu}_{a}$$  

공분산의 형태가 복잡하게 보이지만 Schur complement를 사용하면  

$$(\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})^{-1} = \Sigma_{aa}$$  

임을 알 수 있다. 정리하자면  

- \\(\mathbb{E}[\boldsymbol{x}_{a}] = \boldsymbol{\mu}_{a}\\)
- \\(conv[\boldsymbol{x}_{a}] = \Sigma_{aa}\\)


## 가우시안 분포를 위한 베이즈 정리 (Bayes' Theorem for Gaussian Variables)
$$p(\boldsymbol{x})$$와 $$p(\boldsymbol{y}\vert \boldsymbol{x})$$가 주어져 있고 $$p(\boldsymbol{y}\vert \boldsymbol{x})$$의 평균은 $$\boldsymbol{x}$$의 선형함수이고 공분산은 $$\boldsymbol{x}$$와 독립적이라고 하자. 이제 $$p(\boldsymbol{x})$$와 $$p(\boldsymbol{y}\vert \boldsymbol{x})$$를 구할 것이다. 이 결과는 다음 시간에 배울 선형회귀(베이지안) 주요 내용을 유도하는 데 유용하게 쓰일 것이다.  

$$p(\boldsymbol{x})$$와 $$p(\boldsymbol{y}\vert \boldsymbol{x})$$가 다음과 같이 주어진다고 하자.  

$$p(\boldsymbol{x}) = \mathcal{N}(\boldsymbol{x}\vert \boldsymbol{\mu}, \Lambda^{-1})$$  
$$p(\boldsymbol{y}\vert \boldsymbol{x}) = \mathcal{N}(\boldsymbol{y}\vert \boldsymbol{Ax+b}, \boldsymbol{L}^{-1})$$  

먼저 $$\boldsymbol{z} = \begin{bmatrix} \boldsymbol{x} \cr \boldsymbol{y} \end{bmatrix}$$를 위한 결합확률분포를 구하자. 이 결합확률분포를 구하고 나면 $$p(\boldsymbol{y}$$와 $$p(\boldsymbol{x}\vert \boldsymbol{y})$$는 앞에서 얻은 결과에 의해 쉽게 유도할 수 있다. 먼저 로그값을 생각해보자.  

$$\begin{align}\ln p(\boldsymbol{z}) &= \ln p(\boldsymbol{x}) + \ln p(\boldsymbol{y}\vert \boldsymbol{x}) \\ &= -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{T}\Lambda(\boldsymbol{x}-\boldsymbol{\mu}) -\frac{1}{2}(\boldsymbol{y}-\boldsymbol{Ax}-\boldsymbol{b})^{T}\boldsymbol{L}(\boldsymbol{y}-\boldsymbol{Ax}-\boldsymbol{b})+const   \end{align}$$  

$$\boldsymbol{z}$$의 이차항은 다음과 같다.  

$$\begin{align} -\frac{1}{2}\boldsymbol{x}^{T}(\Lambda + \boldsymbol{A}^{T}\boldsymbol{LA})\boldsymbol{x} - \frac{1}{2}\boldsymbol{y}^{T}\boldsymbol{Ly} + \frac{1}{2}\boldsymbol{y}^{T}\boldsymbol{LAx} + \frac{1}{2}\boldsymbol{x}^{T}\boldsymbol{A}^{T}\boldsymbol{Ly} &= -\frac{1}{2}\begin{bmatrix} \boldsymbol{x} \cr \boldsymbol{y} \end{bmatrix}^{T}\begin{bmatrix} \Lambda + \boldsymbol{A}^{T}\boldsymbol{LA} & -\boldsymbol{A}^{T}\boldsymbol{L} \cr -\boldsymbol{LA} & \boldsymbol{L} \end{bmatrix}\begin{bmatrix} \boldsymbol{x} \cr \boldsymbol{y} \end{bmatrix} \\ &= -\frac{1}{2}\boldsymbol{z}^{T}\boldsymbol{Rz} \end{align}$$  

$$\boldsymbol{R} = \begin{bmatrix} \Lambda + \boldsymbol{A}^{T}\boldsymbol{LA} & -\boldsymbol{A}^{T}\boldsymbol{L} \cr -\boldsymbol{LA} & \boldsymbol{L} \end{bmatrix}$$

따라서 공분산은  

$$cov[\boldsymbol{z}] = \boldsymbol{R}^{-1} = \begin{bmatrix} \Lambda^{-1} & \Lambda^{-1}\boldsymbol{A}^{T} \cr \boldsymbol{A}\Lambda^{-1} & \boldsymbol{L}^{-1}+\boldsymbol{A}\Lambda^{-1}\boldsymbol{A}^{T} \end{bmatrix}$$  

이다.  

평균벡터를 찾기 위해서 $$\boldsymbol{z}$$의 1차항을 정리한다.  

$$\boldsymbol{x}^{T}\Lambda\boldsymbol{\mu} - \boldsymbol{x}^{T}\boldsymbol{A}^{T}\boldsymbol{Lb} + \boldsymbol{y}^{T}\boldsymbol{Lb} = \begin{bmatrix} \boldsymbol{x} \cr \boldsymbol{y} \end{bmatrix}^{T}\begin{bmatrix} \Lambda\boldsymbol{\mu}-\boldsymbol{A}^{T}\boldsymbol{Lb} \cr \boldsymbol{Lb} \end{bmatrix}$$  

따라서 평균벡터는  

$$\mathbb{E}[\boldsymbol{z}] = \boldsymbol{R}^{-1}\begin{bmatrix} \Lambda\boldsymbol{\mu}-\boldsymbol{A}^{T}\boldsymbol{Lb} \cr \boldsymbol{Lb} \end{bmatrix} = \begin{bmatrix} \boldsymbol{\mu} \cr \boldsymbol{A\mu}+\boldsymbol{b} \end{bmatrix}$$  

$$\boldsymbol{y}$$를 위한 주변확률분포의 평균과 공분산은 앞의 "주변 가우시안 분포" 결과를 적용하면 쉽게 구할 수 있다.  

$$\mathbb{E}[\boldsymbol{y}] = \boldsymbol{A\mu} + \boldsymbol{b}$$  
$$cov[\boldsymbol{y}] = \boldsymbol{L}^{-1} + \boldsymbol{A}\Lambda^{-1}\boldsymbol{A}^{T}$$  

마찬가지로 조건부 확률 $$p(\boldsymbol{x}\vert \boldsymbol{y})$$의 평균과 공분산은 "조건부 가우시안 분포" 결과를 적용해 유도할 수 있다.  

$$\mathbb{E}[\boldsymbol{x}\vert \boldsymbol{y}] = (\Lambda + \boldsymbol{A}^{T}\boldsymbol{LA})^{-1}\{ \boldsymbol{A}^{T}\boldsymbol{L}(\boldsymbol{y} - \boldsymbol{b}) + \Lambda\boldsymbol{\mu} \}$$  
$$cov[\boldsymbol{x}\vert \boldsymbol{y}] = (\Lambda + \boldsymbol{A}^{T}\boldsymbol{LA})^{-1}$$  

## 가우시안 분포의 최대우도 (Maximum Likelihood for the Gaussian)
가우시안 분포에 의해 생성된 데이터 $$\boldsymbol{X} = (\boldsymbol{x}_{1},\cdots,\boldsymbol{x}_{n})^{T}$$가 주어졌을 때, 우도를 최대화하는 파리미터 값들(평균, 공분산)을 찾는 것이 목표라고 하자. 로그우도 함수는 다음과 같다.  

$$\ln p(\boldsymbol{X}\vert \boldsymbol{\mu}, \Sigma) = -\frac{ND}{2}\ln(2\pi) -\frac{N}{2}\ln\vert \Sigma\vert - \frac{1}{2}\sum_{n=1}^{N}(\boldsymbol{x}_{n}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\boldsymbol{x}_{n}-\boldsymbol{\mu})$$  

먼저 우도를 최대화하는 평균벡터 $$\boldsymbol{\mu}_{ML}$$을 찾아보자.  

$$\boldsymbol{y} = (\boldsymbol{x}-\boldsymbol{\mu})$$라 하면 다음의 식이 유도된다.  

$$\frac{\partial}{\partial\boldsymbol{\mu}}(\boldsymbol{x}-\boldsymbol{\mu})^{T}\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu}) = \frac{\partial}{\partial\boldsymbol{y}}\boldsymbol{y}^{T}\Sigma^{-1}\boldsymbol{y}\frac{\partial\boldsymbol{y}}{\partial\boldsymbol{\mu}} = -2\Sigma^{-1}\boldsymbol{y} \equiv -2\Lambda\boldsymbol{y}$$  

따라서,  

$$\frac{\partial}{\partial\boldsymbol{\mu}}\ln p(\boldsymbol{X}\vert \boldsymbol{\mu}, \Sigma) = -\frac{1}{2}\sum_{i=1}^{N}-2\Lambda(\boldsymbol{x}_{i}-\boldsymbol{\mu}) = \Lambda\sum_{i=1}^{N}(\boldsymbol{x}_{i}-\boldsymbol{\mu}) = 0$$  

$$\boldsymbol{\mu}_{ML} = \frac{1}{N}\sum_{i=1}^{N}\boldsymbol{x}_{i} = \boldsymbol{\bar x}$$

다음으로 우도를 최대화하는 공분산행렬 $$\Sigma_{ML}$$을 찾아보자.  

$$l(\Lambda) = \frac{N}{2}\ln\vert \Lambda\vert - \frac{1}{2}\sum_{n=1}^{N}tr((\boldsymbol{x}_{n}-\boldsymbol{\mu})(\boldsymbol{x}_{n}-\boldsymbol{\mu})^{T}\Lambda) = \frac{N}{2}\ln\vert \Lambda\vert - \frac{1}{2}tr(\boldsymbol{S}\Lambda)$$  

$$\boldsymbol{S} = \sum_{i=1}^{N}(\boldsymbol{x}_{n}-\boldsymbol{\mu})(\boldsymbol{x}_{n}-\boldsymbol{\mu})^{T}$$  

$$\frac{\partial l(\Lambda)}{\partial \Lambda} = \frac{N}{2}(\Lambda^{-1})^{T}-\frac{1}{2}\boldsymbol{S}^{T} = 0$$  

$$(\Lambda_{ML}^{-1}) = \Sigma_{ML} = \frac{1}{N}\boldsymbol{S}$$  

$$\Sigma_{ML} = \frac{1}{N}\sum_{i=1}^{N}(\boldsymbol{x}_{n}-\boldsymbol{\mu})(\boldsymbol{x}_{n}-\boldsymbol{\mu})^{T}$$  


위의 식 유도를 위해 아래의 기본벅인 선형대수 결과를 사용하였다.  

- \\(\vert \boldsymbol{A}^{-1}\vert = l/\vert \boldsymbol{A}\vert\\)
- \\(\boldsymbol{x}^{T}\boldsymbol{Ax} = tr(\boldsymbol{x}^{T}\boldsymbol{Ax}) = tr(\boldsymbol{xx}^{T}\boldsymbol{A}) \\)
- \\(tr(\boldsymbol{A}) + tr(\boldsymbol{B}) = tr(\boldsymbol{A}+\boldsymbol{B}) \\)
- \\(\frac{\partial}{\partial\boldsymbol{A}}tr(\boldsymbol{BA}) = \boldsymbol{B}^{T} \\)
- \\(\frac{\partial}{\partial\boldsymbol{A}}\ln\vert \boldsymbol{A}\vert = (\boldsymbol{A}^{-1})^{T}\\)

$$(\Lambda_{ML})^{-1} = \Sigma_{ML}$$이해하기  

일반적으로 다음이 성립한다. 함수 $$h(\boldsymbol{X}) = \boldsymbol{Y}$$가 일대일이고 다음과 같은 최소값들이 존재한다고 하자.  

$$\boldsymbol{X}^{*} = \arg\min_{X}f(h(\boldsymbol{X}))$$  

$$\boldsymbol{Y}^{*} = \arg\min_{Y}f(\boldsymbol{Y})$$  

$$f(h(h^{-1}(\boldsymbol{Y}^{*}))) = f(\boldsymbol{Y}^{*})$$이므로 $$h^{-1}(\boldsymbol{Y}^{*}) = \boldsymbol{X}^{*}$$이 성립한다. 위의 경우에 적용하자면,  

$$\Lambda_{ML} = \arg\min_{\Lambda}l(\Lambda)$$  

역행렬 연산이 일대일함수$$(h)$$를 정의하기 때문에, $$h^{-1}(\Lambda_{ML}) = \Sigma_{ML}$$이 되고, $$(\Lambda_{ML})^{-1} = \Sigma_{ML}$$이 성립한다.  

## 가우시안 분포를 위한 베이지안 추론 (Bayesian Inference for the Gaussian)
MLE 방법은 파라미터들 ($$\boldsymbol{\mu}, \Sigma$$)의 하나의 값만을 구하게 해준다. 베이지안 방법을 사용하면 파라미터의 확률분포 자체를 구할 수 있게 된다.  
단변량 가우시안 확률변수 $$x$$의 $$\mu$$를 베이지안 추론을 통해 구해보자(분산 $$\sigma^2$$는 주어졌다고 가정). 목표는 $$\mu$$의 사후확률 $$p(\mu\vert \boldsymbol{X})$$을 우도함수 $$p(\boldsymbol{X}\vert \mu)$$와 사전확률 $$p(\boldsymbol{X}\vert \mu)$$을 통해 구하는 것이다.  

- 우도함수  

$$p(\boldsymbol{x}\vert \mu) = \prod_{n=1}^{N}p(\boldsymbol{x}_{n}\vert \mu) = \frac{1}{(2\pi\sigma^2)^{N/2}}\exp\left\{-\frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_{n}-\mu)^2 \right\}$$  

- 사전확률  

$$p(\mu) = \mathcal{N}(\mu\vert \mu_{0}, \sigma_{0}^{2})$$  

- 사후확률  

$$\begin{align}p(\mu\vert x) &= \mathcal{N}(\mu\vert \mu_{N}, \sigma_{N}^{2}) \\\\ \mu_{N} &= \frac{\sigma^2}{N\sigma_{0}^2+\sigma^2}\mu_{0}+\frac{N\sigma_{0}^2}{N\sigma_{0}^2+\sigma^2}\mu_{ML} \\\\ \frac{1}{\sigma_{N}^{2}} &= \frac{1}{\sigma_{0}^{2}} + \frac{N}{\sigma^2}  \\\\ \mu_{ML} &= \frac{1}{N}\sum_{n=1}^{N}x_{n} \end{align}$$  

![Dirichlet-normalize-1](/assets/images/Dirichlet-normalize-1.png){: .align-center}  

# Appendix
## Minimal-mistakes
Video Embedding for youtube:  
```
{% include video id="XsxDH4HcOWA" provider="youtube" %}
```  

> other video providers: <https://mmistakes.github.io/minimal-mistakes/docs/helpers/>
## MathJax

\setminus  
$$\{A\}\setminus\{B\}$$:  
```
$$\{A\}\setminus\{B\}$$
```
adjustable parenthesis  
$$\left( content \right)$$:
```
$$\left( content \right)$$
```

Matrix with parenthesis $$\begin{pmatrix}1 \cr 1 \end{pmatrix}$$:  
```
$$\begin{pmatrix}1 \cr 1 \end{pmatrix}$$
```
## References

> Pattern Recognition and Machine Learning: <https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf>  
