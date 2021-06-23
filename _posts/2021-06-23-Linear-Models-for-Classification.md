---
date: 2021-06-23 15:52
title: "ML basics - Linear Models for Classification"
categories: DevCourse2 Classification MathJax
tags: DevCourse2 Classification MathJax
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---


# 선형분류의 목표와 방법들
분류(classification)의 목표

- 입력벡터 $$\boldsymbol{x}$$를 $$K$$개의 가능한 클래스 중에서 하나의 클래스로 할당하는 것

분류를 위한 결정이론

- 확률적 모델 (probabilistic model)
    -생성모델 (generative model): $$p(\boldsymbol{x}\vert \mathcal{C}_k)$$와 $$p(\mathcal{C}_k)$$를 모델링한다음 베이즈 정리를 사용해서 클래스의 사후 확률 $$p(\mathcal{C}_k\vert \boldsymbol{x})$$를 구한다. 또는 결합확률 $$p(\boldsymbol{x}, \mathcal{C}_k)$$을 직접 모델링할 수도 있다.
    - 식별모델 (discriminative model): $$p(\mathcal{C}_k\vert \boldsymbol{x})$$를 직접적으로 모델링한다.
- 판별함수 (discriminant function): 입력 $$\boldsymbol{x}$$을 클래스로 할당하는 판별함수(discriminant function)를 찾는다. 확률값은 계산하지 않는다.

# 판별함수 (Discriminant functions)

입력 $$\boldsymbol{x}$$을 클래스로 할당하는 판별함수(discriminant function)를 찾고자 한다. 여기서는 그러한 함수 중 선형함수만을 다룰 것이다.
두 개의 클래스

선형판별함수는 다음과 같다.  

$$y(\boldsymbol{x})=\boldsymbol{w}^T\boldsymbol{x}+w_0$$  

\- $$\boldsymbol{w}$$: 가중치 벡터 (weight vector)  
\- $$w_0$$: 바이어스 (bias)

$$y(\boldsymbol{x}) \ge 0$$ 인 경우 이를 $$\mathcal{C}_1$$으로 판별하고 아닌 경우 $$\mathcal{C}_2$$으로 판별한다.

결정 경계 (decision boundary)

\- $$y(\boldsymbol{x})=0$$  
\- $$D-1$$차원의 hyperplane ($$\boldsymbol{x}$$가 $$D$$차원의 입력벡터일 때)  

> $$y(\boldsymbol{x})=0$$을 만족시키는 $$\boldsymbol{x}$$의 집합을 결정 경계라고 함.  

결정 경계면 위의 임의의 두 점 $$\boldsymbol{x}_A$$와 $$\boldsymbol{x}_B$$

\- $$y(\boldsymbol{x}_A)=y(\boldsymbol{x}_B)=0$$  
\- $$\boldsymbol{w}^T(\boldsymbol{x}_A - \boldsymbol{x}_B)=0$$ => $$\boldsymbol{w}$$는 결정 경계면에 수직  

> 임의의 두 점 $$\boldsymbol{x}_A$$와 $$\boldsymbol{x}_B$$에 대해서 위의 식이 성립하기 때문에, Decision Boundary에 있는 모든 두 개의 점에 대해서 벡터 $$\boldsymbol{w}$$는 결정 경게면에 수직임.  

원점에서 결정경계면까지의 거리

벡터 $$\boldsymbol{w}_{\perp}$$를 원점에서 결정 경계면에 대한 사영(projection)이라고 하자.   

> $$\boldsymbol{w}$$의 단위벡터 ($$\frac{\boldsymbol{w}}{\Vert  \boldsymbol{w}\Vert }$$)에 $$r$$을 곱했을때 projection vector ($$\boldsymbol{w}_{\perp}$$)가 됨. 이 r를 구하면 결정경계와 원점의 거리가 됨.  

> projection vector ($$\boldsymbol{w}_{\perp}$$)가 결정경계 위에 있기 때문에 $$y(\boldsymbol{w}_{\perp}) = 0$$가 됨.  

> $$\boldsymbol{w}^T\boldsymbol{w}$$은 l2 norm의 제곱 ($$\Vert \boldsymbol{w}\Vert_{2}^{2}$$)이다.

$$\begin{align} &\ r\frac{\boldsymbol{w}}{\Vert  \boldsymbol{w}\Vert } = \boldsymbol{w}_{\perp}\\ &\ y(\boldsymbol{w}_{\perp}) = 0\\ &\ \boldsymbol{w}^T\boldsymbol{w}_{\perp} + w_0 = 0\\ &\ \frac{\boldsymbol{w}^T\boldsymbol{w}}{\Vert \boldsymbol{w}\Vert }r + w_0 = 0\\ &\ \Vert \boldsymbol{w}\Vert  r + w_0 = 0\\ &\  r = -\frac{w_0}{\Vert \boldsymbol{w}\Vert } \end{align}$$

따라서 $$w_0$$은 결정 경계면의 위치를 결정한다.

\- $$w_0\lt0$$이면 결정 경계면은 **원점으로부터** $$\boldsymbol{w}$$가 향하는 방향으로 멀어져있다.  
\- $$w_0\gt0$$이면 결정 경계면은 **원점으로부터** $$\boldsymbol{w}$$의 반대 방향으로 멀어져있다.  

> 여기서 중요한 부분은 결정 경계의 위치가 어디에 있느냐임. 헷갈릴 수도 있지만. bias 값이 결정경계면의 위치를 결정하는 파라미터라는 것임.   

![w-and-decision-boundary](/assets/images/w-and-decision-boundary.png){: .align-center .img-50}  

예제: $$y(x_1, x_2) = x_1 + x_2 - 1$$

또한 $$y(\boldsymbol{x})$$값은 $$\boldsymbol{x}$$와 결정 경계면 사이의 **부호화된 거리**와 비례한다.  

> $$\boldsymbol{x}-\boldsymbol{x}_\perp$$이 벡터는 $$\boldsymbol{w}$$와 방향이 같지만 길이는 r인 단위벡터임. 이것을 다시 써보면 $$\boldsymbol{x}-\boldsymbol{x}_\perp = r\frac{\boldsymbol{w}}{\Vert \boldsymbol{w}\Vert }$$라는 말임.  

> $$\boldsymbol{x} = \boldsymbol{x}_\perp + r\frac{\boldsymbol{w}}{\Vert \boldsymbol{w}\Vert }$$ 여기에 $$\boldsymbol{w}^{T}$$를 곱하고 bias인 $$w_{0}$$을 더해줌.  

> $$\boldsymbol{w}^{T}\boldsymbol{x}+w_{0} = \boldsymbol{w}^{T}\boldsymbol{x}_\perp + w_{0} + r\frac{\boldsymbol{w}^{T}\boldsymbol{w}}{\Vert \boldsymbol{w}\Vert }$$  

> $$\boldsymbol{x}_\perp$$이 결정경계 위에 있기 때문에 $$\boldsymbol{x}_\perp + w_{0}$$ 이 부분이 0이 됨. 또한 $$\frac{\boldsymbol{w}^{T}\boldsymbol{w}}{\Vert \boldsymbol{w}\Vert }$$은 $$\Vert \boldsymbol{w}\Vert$$이 됨. 왼쪽에 있는 항은 $$y(\boldsymbol{x})$$이 됨. 

> 따라서 $$y(\boldsymbol{x}) = r\Vert \boldsymbol{w}\Vert$$이 됨. $$r$$ 곱하기 $$\boldsymbol{w}$$의 l2 norm.  

> 다시 정리하면, $$r=\frac{y(\boldsymbol{x})}{\Vert \boldsymbol{w}\Vert }$$이 되는 것임.  

임의의 한 점 $$\boldsymbol{x}$$의 결정 경계면에 대한 사영을 $$\boldsymbol{x}_\perp$$이라고 하자.  

$$\boldsymbol{x}=\boldsymbol{x}_\perp + r\frac{\boldsymbol{w}}{\Vert \boldsymbol{w}\Vert }$$  

$$r=\frac{y(\boldsymbol{x})}{\Vert \boldsymbol{w}\Vert }$$  

> 좀 전에는 bias에 의해 결정경계면의 위치가 결정되었다면, 이번에는 $$y(\boldsymbol{x})$$에 의해 $$\boldsymbol{x}$$의 위치가 결정됨.  

\- $$y(\boldsymbol{x}) \gt 0$$이면 $$\boldsymbol{x}$$는 결정 경계면을 기준으로 $$\boldsymbol{w}$$가 향하는 방향에 있다.  
\- $$y(\boldsymbol{x}) \lt 0$$이면 $$\boldsymbol{x}$$는 결정 경계면을 기준으로 $$-\boldsymbol{w}$$가 향하는 방향에 있다.  
\- $$y(\boldsymbol{x})$$의 절대값이 클 수록 더 멀리 떨어져 있다.

가짜입력(dummy input) $$x_0=1$$을 이용해서 수식을 단순화  

> dummy input을 사용했다는 의미로 tilde 심볼을 사용함.  

\- $$\widetilde{\boldsymbol{w}}=(w_0, \boldsymbol{w})$$  
\- $$\widetilde{\boldsymbol{x}}=(x_0, \boldsymbol{x})$$  
\- $$y(\boldsymbol{x})=\boldsymbol{\widetilde{w}}^T\boldsymbol{\widetilde{x}}$$  

![geometry-of-a-linear-discriminant-function](/assets/images/geometry-of-a-linear-discriminant-function.png){: .align-center}  

## 다수의 클래스  

$$y_k(\boldsymbol{x})=\boldsymbol{w}_k^T\boldsymbol{x}+w_{k0}$$  

> 각각의 클래스 $$k$$ 마다 weight vector인 $$\boldsymbol{w}_k$$를 학습해야 함.  

$$k=1,\ldots,K$$  

위와 같은 판별함수는 $$j{\neq}k$$일 때 $$y_k(\boldsymbol{x})\gt y_j(\boldsymbol{x})$$를 만족하면 $$\boldsymbol{x}$$를 클래스 $$\mathcal{C}_k$$로 판별하게 된다.  

# 분류를 위한 최소제곱법 (Least squares for classification)

> 그렇다면 파라미터 $$\boldsymbol{w}$$를 어떻게 학습할 수 있을까? 간단하게 하는 법은 최소제곱법. 선형회귀에서는 목표값이 실수값으로 주어졌기 때문에, 자연스럽게 될 수 있지만, 분류에서는 실수값이긴 하지만, 만약에 2개의 클래스라면 0 or 1 으로 실수값의 목표값으로 변환시키는 방식으로 함.  

> 결론적으로는 분류에는 별로 좋지 않은 방식이다.  

$$y_k(\boldsymbol{x})=\boldsymbol{w}_k^T\boldsymbol{x}+w_{k0}$$  

$$k=1,\ldots,K$$  

아래와 같이 행렬 $$\widetilde{\boldsymbol{W}}$$을 사용하여 간편하게 나타낼 수 있다.  

$$y(\boldsymbol{x}) = \widetilde{\boldsymbol{W}}^T\widetilde{\boldsymbol{x}}$$  

> $$~~ \widetilde{\boldsymbol{W}} = \begin{bmatrix} \vert\\ \cdots \widetilde{\boldsymbol{w}}_k \cdots\\ \vert \end{bmatrix}$$인데, K=3인 경우, $$y_1(\boldsymbol{x})=\widetilde{\boldsymbol{w}}_{1}^T\widetilde{\boldsymbol{x}}$$, $$y_2(\boldsymbol{x})=\widetilde{\boldsymbol{w}}_{2}^T\widetilde{\boldsymbol{x}}$$, $$y_3(\boldsymbol{x})=\widetilde{\boldsymbol{w}}_{3}^T\widetilde{\boldsymbol{x}}$$ 이 각각의 값들은 scalar값인데 이 것을 하나의 표현할 것임.  

> $$y(\boldsymbol{x}) = \begin{bmatrix} \widetilde{\boldsymbol{w}}_{1}^T\widetilde{\boldsymbol{x}}\\ \widetilde{\boldsymbol{w}}_{2}^T\widetilde{\boldsymbol{x}}\\ \widetilde{\boldsymbol{w}}_{3}^T\widetilde{\boldsymbol{x}}\\ \end{bmatrix} = \begin{bmatrix} - \widetilde{\boldsymbol{w}}_{1}^T - \\ - \widetilde{\boldsymbol{w}}_{2}^T - \\ - \widetilde{\boldsymbol{w}}_{3}^T - \\ \end{bmatrix}\widetilde{\boldsymbol{x}} = \widetilde{\boldsymbol{W}}^T\widetilde{\boldsymbol{x}}$$ 이런식으로 표현되는 것임.  

$$\widetilde{\boldsymbol{W}}$$의 $$k$$번째 열은 $$\widetilde{\boldsymbol{w}}_k = (w_{k0}, \boldsymbol{w}_k^T)^T$$이다.

## 제곱합 에러 함수

학습데이터 $$\{\boldsymbol{x}_n, \boldsymbol {T}_n\}$$, $$n=1,\ldots,N$$, $$n$$번째 행이 $$\boldsymbol {T}_n^T$$인 행렬 $$\boldsymbol {T}$$, $$n$$번째 행이 $$\widetilde{\boldsymbol{x}}_n^T$$인 행렬 $$\widetilde{\boldsymbol{X}}$$이 주어졌을 때 제곱합 에러함수(sum-of-squared error function)은  

$$E_D(\widetilde{\boldsymbol{W}}) = \frac{1}{2}\mathrm{tr}\left\{ \left(\widetilde{\boldsymbol{X}}\widetilde{\boldsymbol{W}}-\boldsymbol{T} \right)^T \left(\widetilde{\boldsymbol{X}}\widetilde{\boldsymbol{W}}-\boldsymbol{T} \right) \right\}$$  

> 유의할점은 K>2 일 경우 \widetilde{\boldsymbol{W}}가 행렬이라는 점이다. 각각의 열이 하나의 클래스에 대응함.  

> 선형대수에서 했던 PCA와 유사함  

> 다른식으로 정리하면, $$E_D(\widetilde{\boldsymbol{W}}) = \frac{1}{2}\Vert\widetilde{\boldsymbol{X}}\widetilde{\boldsymbol{W}} \Vert_{F}^{2}$$: 행렬과 행렬의 차이에 Frobenius norm을 제곱한 것.(식을 간편하게 하기 위해 1/2 곱해줌) 이것이 $$E_D(\widetilde{\boldsymbol{W}}) = \frac{1}{2}\mathrm{tr}\left\{ \left(\widetilde{\boldsymbol{X}}\widetilde{\boldsymbol{W}}-\boldsymbol{T} \right)^T \left(\widetilde{\boldsymbol{X}}\widetilde{\boldsymbol{W}}-\boldsymbol{T} \right) \right\}$$ 이 것과 동일함.  

> <https://marquis08.github.io/devcourse2/linearalgebra/mathjax/ML-basics-Linear-Algebra/#%EB%8C%80%EA%B0%81%ED%95%A9-trace>  

> 제곱합 에러를 생각할때, 행렬의 Frobenius norm을 최소화 시킨다라는 것을 생각.

로 표현할 수 있다.  

아래와 같이 유도할 수 있다. **Design Matrix**  

$$\widetilde{\boldsymbol{X}} = \begin{bmatrix} \vdots\\ - \widetilde{\boldsymbol{x}}_n^T -\\ \vdots \end{bmatrix},~~ \widetilde{\boldsymbol{W}} = \begin{bmatrix} \vert\\ \cdots \widetilde{\boldsymbol{w}}_k \cdots\\ \vert \end{bmatrix},~~ \boldsymbol {T}= \begin{bmatrix} \vdots\\ - \boldsymbol {T}_n^T -\\ \vdots \end{bmatrix}$$  

> $${T}= \begin{bmatrix} \vdots\\ - \boldsymbol {T}_n^T -\\ \vdots \end{bmatrix}$$는 각각의 행은 K개의 값들을 가진 벡터임.  

$$\begin{align} E_D(\widetilde{\boldsymbol{W}}) &\ = \frac{1}{2}\sum_{n=1}^N\sum_{k=1}^K \left(\widetilde{\boldsymbol{x}}_n^T\widetilde{\boldsymbol{w}}_k - \boldsymbol {T}_{nk}\right)^2\\ &\ = \frac{1}{2}\sum_{n=1}^N \left( \widetilde{\boldsymbol{x}}_n^T \widetilde{\boldsymbol{W}} - \boldsymbol {T}_n^T \right) \left( \widetilde{\boldsymbol{x}}_n^T \widetilde{\boldsymbol{W}} - \boldsymbol {T}_n^T \right)^T\\ &\ = \frac{1}{2}\sum_{n=1}^N \mathrm{tr}\left\{ \left( \widetilde{\boldsymbol{x}}_n^T \widetilde{\boldsymbol{W}} - \boldsymbol {T}_n^T \right) \left( \widetilde{\boldsymbol{x}}_n^T \widetilde{\boldsymbol{W}} - \boldsymbol {T}_n^T \right)^T \right\}\\ &\ = \frac{1}{2}\sum_{n=1}^N \mathrm{tr}\left\{ \left( \widetilde{\boldsymbol{x}}_n^T \widetilde{\boldsymbol{W}} - \boldsymbol {T}_n^T \right)^T \left( \widetilde{\boldsymbol{x}}_n^T \widetilde{\boldsymbol{W}} - \boldsymbol {T}_n^T \right) \right\}\\ &\ = \frac{1}{2}\mathrm{tr}\left\{ \sum_{n=1}^N \left( \widetilde{\boldsymbol{x}}_n^T \widetilde{\boldsymbol{W}} - \boldsymbol {T}_n^T \right)^T \left( \widetilde{\boldsymbol{x}}_n^T \widetilde{\boldsymbol{W}} - \boldsymbol {T}_n^T \right) \right\}\\ &\ = \frac{1}{2}\mathrm{tr}\left\{ \left(\widetilde{\boldsymbol{x}}\widetilde{\boldsymbol{W}}-\boldsymbol {T} \right)^T \left(\widetilde{\boldsymbol{x}}\widetilde{\boldsymbol{W}}-\boldsymbol {T} \right) \right\} \end{align}$$  

마지막 과정

> 이 부분도 PCA 할대 했던 것임.  

$$\begin{align} {\boldsymbol A} &= \boldsymbol{X}\boldsymbol{W} - \boldsymbol{T} = \begin{bmatrix} \vdots\\ - \boldsymbol{x}_n^T \boldsymbol{W} - \boldsymbol {T}_n^T-\\ \vdots \end{bmatrix}\\ {\boldsymbol A}^T{\boldsymbol A} &= \begin{bmatrix} \vert\\ \cdots \left(\boldsymbol{x}_n^T \boldsymbol{W} - \boldsymbol {T}_n^T\right)^T\cdots\\ \vert \end{bmatrix} \begin{bmatrix} \vdots\\ - \boldsymbol{x}_n^T \boldsymbol{W} - \boldsymbol {T}_n^T-\\ \vdots \end{bmatrix}\\ &= \sum_{n=1}^N \left( \boldsymbol{x}_n^T \boldsymbol{W} - \boldsymbol {T}_n^T \right)^T \left( \boldsymbol{x}_n^T \boldsymbol{W} - \boldsymbol {T}_n^T \right) \end{align}$$  

$$\widetilde{\boldsymbol{W}}$$에 대한 $$E_D(\widetilde{\boldsymbol{W}})$$의 최솟값을 구하면  

$$\widetilde{\boldsymbol{W}}=(\widetilde{\boldsymbol{X}}^T\widetilde{\boldsymbol{X}})^{-1}\widetilde{\boldsymbol{X}}^T\boldsymbol {T}=\widetilde{\boldsymbol{X}}^{\dagger}\boldsymbol{T}$$  

> $$\widetilde{\boldsymbol{X}}^{\dagger}$$ $$\boldsymbol{X}$$의 pseudo inverse에 목표값 행렬을 곱한 것이 에러함수를 최소화 시키는 $$\boldsymbol{W}$$임.

따라서 판별함수는 다음과 같다.  

$$y(\boldsymbol{x})=\widetilde{\boldsymbol{W}}^T\widetilde{\boldsymbol{x}}=\boldsymbol{T}^T\left(\widetilde{\boldsymbol{X}}^{\dagger}\right)^T\widetilde{\boldsymbol{x}}$$  

- 분류를 위한 최소제곱법의 문제들
    - 극단치에 민감하다.
    - 목표값의 확률분포에 대한 잘못된 가정에 기초해 있다.  

![problems-of-least-squares-1](/assets/images/problems-of-least-squares-1.png){: .align-center}  
![problems-of-least-squares-2](/assets/images/problems-of-least-squares-2.png){: .align-center}  

# 퍼셉트론 알고리즘 (The perceptron algorithm)
$$y(\boldsymbol{x})=f(\boldsymbol{w}^T\phi(\boldsymbol{x}))$$  

여기서 $$f$$는 활성 함수(activation fucntion)로 퍼셉트론은 아래와 같은 계단형 함수를 사용한다.  

$$f(a)= \left\{ {\begin{array}{ll}+1, &\  a \ge 0 \\-1, &\  a \lt 0 \end{array}} \right.$$  

여기서 $$\phi_0(\boldsymbol{x})=1$$이다.  

에러함수  

$$E_P(\boldsymbol{w})=\sum_{n \in \mathcal{M}}\boldsymbol{w}^T\phi_nt_n$$  

$$\mathcal{M}$$은 잘못 분류된 데이터들의 집합  

Stochastic gradient descent의 적용  

$$\boldsymbol{w}^{(\tau+1)}=\boldsymbol{w}^{(\tau)}\eta\triangledown E_p(\boldsymbol{w})=\boldsymbol{w}^{(\tau)}+\eta\phi_n{t_n}$$

위 업데이트가 실행될 때 잘못 분류된 샘플에 미치는 영향  

$$\boldsymbol{w}^{(\tau+1)T}{\phi}_n{t_n} = \boldsymbol{w}^{(\tau)T}{\phi_n}{t_n}-(\phi_n{t_n})^T\phi_n{t_n} \lt \boldsymbol{w}^{(\tau)T}\phi_n{t_n}$$  

![affects-of-misclassified-samples-1](/assets/images/affects-of-misclassified-samples-1.png){: .align-center}
![affects-of-misclassified-samples-2](/assets/images/affects-of-misclassified-samples-2.png){: .align-center}
![affects-of-misclassified-samples-3](/assets/images/affects-of-misclassified-samples-3.png){: .align-center}  

# 확률적 생성 모델 (Probabilistic Generative Models)

이제 분류문제를 확률적 관점에서 살펴보고자 한다. 선형회귀와 마찬가지로 확률적 모델은 통합적인 관점을 제공해준다. 예를 들어 데이터의 분포에 관해 어떤 가정을 두게 되면 앞에서 살펴본 선형적인 결정경계(linear decision boundary)가 그 결과로 유도되는 것을 보게 될 것이다.

$$p(\boldsymbol{x}\vert \mathcal{C}_k)$$와 $$p(\mathcal{C}_k)$$를 모델링한다음 베이즈 정리를 사용해서 클래스의 사후 확률 $$p(\mathcal{C}_k\vert \boldsymbol{x})$$를 구한다. 이전의 판별함수 방법에서는 어떤 에러함수를 최소화시키기 위한 최적의 파라미터를 찾는 것이 목적이라면 확률적 모델은 데이터의 분포(클래스를 포함한)를 모델링하면서 분류문제를 결과적으로 풀게 된다.  

$$p(\mathcal{C}_1\vert \boldsymbol{x}) = \frac{p(\boldsymbol{x}\vert \mathcal{C}_1)p(\mathcal{C}_1)}{p(\boldsymbol{x}\vert \mathcal{C}_1)p(\mathcal{C}_1)+p(\boldsymbol{x}\vert \mathcal{C}_2)p(\mathcal{C}_2)}=\frac{1}{1+\exp(-a)} = \sigma(a)$$  

$$a=\ln{\frac{p(\boldsymbol{x}\vert \mathcal{C}_1)p(\mathcal{C}_1)}{p(\boldsymbol{x}\vert \mathcal{C}_2)p(\mathcal{C}_2)}}$$  

$$\sigma(a)=\frac{1}{1+\exp(-a)}$$

Logistic sigmoid의 성질 및 역함수

\- $$\sigma(-a) = 1 - \sigma(a)$$  
\- $$a=\ln\left(\frac{\sigma}{1\sigma}\right)$$

$$K\gt2$$인 경우  

$$p(\mathcal{C}_k\vert \boldsymbol{x}) = \frac{p(\boldsymbol{x}\vert \mathcal{C}_k)p(\mathcal{C}_k)}{\sum_j{p(\boldsymbol{x}\vert \mathcal{C}_j)p(\mathcal{C}_j)}}=\frac{\exp(a_k)}{\sum_j{\exp(a_j)}}$$  

$$a_k = p(\boldsymbol{x}\vert \mathcal{C}_k)p(\mathcal{C}_k)$$  

## 연속적 입력 (continous inputs)

$$p(\boldsymbol{x}\vert \mathcal{C}_k)$$가 가우시안 분포를 따르고 모든 클래스에 대해 공분산이 동일하다고 가정하자.  

$$p(\boldsymbol{x}\vert \mathcal{C}_k) = \frac{1}{(2\pi)^{D/2}\vert \Sigma\vert ^{1/2}}\exp\left\{\frac{1}{2}(\boldsymbol{x}-{\pmb \mu}_k)^T\Sigma^{-1}(\boldsymbol{x}-{\pmb \mu}_k)\right\}$$  


두 개의 클래스인 경우  

$$p(\mathcal{C}_1\vert \boldsymbol{x}) = \sigma(a)$$  

$$a$$를 전개하면  

$$\begin{align} a &\ = \ln{\frac{p(\boldsymbol{x}\vert \mathcal{C}_1)p(\mathcal{C}_1)}{p(\boldsymbol{x}\vert \mathcal{C}_2)p(\mathcal{C}_2)}}\\ &\ = \frac{1}{2}(\boldsymbol{x}-{\pmb \mu}_1)^T\Sigma^{-1}(\boldsymbol{x}-{\pmb \mu}_1)+\frac{1}{2}(\boldsymbol{x}-{\pmb \mu}_2)^T\Sigma^{-1}(\boldsymbol{x}-{\pmb \mu}_2)+\ln\frac{p(\mathcal{C}_1)}{p(\mathcal{C}_2)}\\ &\ = \left\{\left( {\pmb \mu}_1^T - {\pmb \mu}_2^T \right)\Sigma^{-1}\right\}\boldsymbol{x} - \frac{1}{2}{\pmb \mu}_1^T\Sigma^{-1}{\pmb \mu}_1 + \frac{1}{2}{\pmb \mu}_2^T\Sigma^{-1}{\pmb \mu}_2 + \ln\frac{p(\mathcal{C}_1)}{p(\mathcal{C}_2)} \end{align}$$  

따라서 $$a$$를 $$\boldsymbol{x}$$에 관한 선형식으로 다음과 같이 정리할 수 있다.  

$$p(\mathcal{C}_1\vert \boldsymbol{x}) = \sigma(\boldsymbol{w}^T\boldsymbol{x}+w_0)$$  

$$\begin{align} \boldsymbol{w} &\ = \Sigma^{-1}({\pmb \mu}_1 - {\pmb \mu}_2)\\ w_0 &\ = - \frac{1}{2}{\pmb \mu}_1^T\Sigma^{-1}{\pmb \mu}_1 + \frac{1}{2}{\pmb \mu}_2^T\Sigma^{-1}{\pmb \mu}_2 + \ln\frac{p(\mathcal{C}_1)}{p(\mathcal{C}_2)} \end{align}$$  

$$K$$개의 클래스인 경우  

$$\boldsymbol{w}_k = \Sigma^{-1}{\pmb \mu}_k$$  

$$w_{k0} = \frac{1}{2}{\pmb \mu}_{k}^{T}\Sigma^{-1}{\pmb \mu}_k + \ln p(\mathcal{C}_k)$$  

## 최대우도해 (Maximum likelihood solution)

이제 MLE를 통해 모델 파라미터들을 구해보자. 두 개의 클래스인 경우를 살펴본다.

데이터

\- $$\{\boldsymbol{x}_n, t_n\}$$, $$n=1,\ldots,N$$. $$t_n=1$$은 클래스 $$\mathcal{C}_1$$을 $$t_n=0$$은 클래스 $$\mathcal{C}_2$$를 나타낸다고 하자.

파라미터들

\- $$p(\mathcal{C}_1)=\pi$$라고 두면, 구해야 할 파라미터들은 $${\pmb \mu}_1$$, $${\pmb \mu}_2$$, $$\Sigma$$, $$\pi$$이다.

우도식 유도

\- $$t_n=1$$이면  

$$p(\boldsymbol{x}_n, \mathcal{C}_1) = p(\mathcal{C}_1)p(\boldsymbol{x}_n\vert \mathcal{C}_1) = \pi \mathcal{N}(\boldsymbol{x}_n\vert \mu_1, \Sigma)$$  

\- $$t_n=0$$이면  

$$p(\boldsymbol{x}_n, \mathcal{C}_2) = p(\mathcal{C}_2)p(\boldsymbol{x}_n\vert \mathcal{C}_2) = (1 - \pi) \mathcal{N}(\boldsymbol{x}_n\vert \mu_2, \Sigma)$$  

따라서 우도함수는  

$$p(\boldsymbol {t}\vert  \pi, {\pmb \mu}_1, {\pmb \mu}_2, \Sigma) = \prod_{n=1}^N\left[\pi \mathcal{N}(\boldsymbol{x}_n\vert {\pmb \mu}_1, \Sigma)\right]^{t_n}\left[(1 - \pi)\mathcal{N}(\boldsymbol{x}_n\vert {\pmb \mu}_2, \Sigma)\right]^{1-t_n}$$  

$$\boldsymbol {t} = (t_1,\ldots,t_N)^T$$  

## $$\pi$$ 구하기

로그우도함수에서 $$\pi$$ 관련항들만 모으면  

$$\sum_{n=1}^{N}\left\{ t_n\ln\pi + (1-t_n)\ln(1\pi) \right\}$$  

이 식을 $$\pi$$에 관해 미분하고 0으로 놓고 풀면  

$$\pi = \frac{1}{N}\sum_{n=1}^{N}t_n = \frac{N_1}{N} = \frac{N_1}{N_1+N_2}$$  

$$N_1$$은 $$\mathcal{C}_1$$에 속하는 샘플의 수이고 $$N_2$$는 $$\mathcal{C}_2$$에 속하는 샘플의 수이다.  

## $${\pmb \mu}_1$$, $${\pmb \mu}_2$$ 구하기

$${\pmb \mu}_1$$ 관련항들  

$$\sum_{n=1}^{N}t_n\ln \mathcal{N}(\boldsymbol{x}_n\vert {\pmb \mu}_1, \Sigma) = \frac{1}{2}\sum_{n=1}^{N}t_n(\boldsymbol{x}_n-{\pmb \mu}_1)^T\Sigma^{-1}(\boldsymbol{x}_n-{\pmb \mu}_1) + \mathrm{const}$$  

이 식을 $${\pmb \mu}_1$$에 관해 미분하고 0으로 놓고 풀면  

$${\pmb \mu}_1=\frac{1}{N_1}\sum_{n=1}^{N}t_n\boldsymbol{x}_n$$  

유사하게  

$${\pmb \mu}_2=\frac{1}{N_2}\sum_{n=1}^{N}(1-t_n)\boldsymbol{x}_n$$  

## $$\Sigma$$ 구하기

$$\begin{align} &\ \frac{1}{2}\sum_{n=1}^{N}t_n\ln \vert \Sigma\vert  \frac{1}{2}\sum_{n=1}^{N}t_n(\boldsymbol{x}_n-{\pmb \mu}_1)^T\Sigma^{-1}(\boldsymbol{x}_n-{\pmb \mu}_1)\\ &\ \frac{1}{2}\sum_{n=1}^{N}(1-t_n)\ln \vert \Sigma\vert  \frac{1}{2}\sum_{n=1}^{N}(1-t_n)(\boldsymbol{x}_n-{\pmb \mu}_2)^T\Sigma^{-1}(\boldsymbol{x}_n-{\pmb \mu}_2)\\ &\ = \frac{N}{2}\ln \vert \Sigma\vert  - \frac{N}{2}\mathrm{tr}\left(\Sigma^{-1}{\boldsymbol S}\right) \end{align}$$  

$$\begin{align} {\boldsymbol S} &\ =\frac{N_1}{N}{\boldsymbol S}_1+\frac{N_2}{N}{\boldsymbol S}_2\\ {\boldsymbol S}_1 &\ = \frac{1}{N_1}\sum_{n \in \mathcal{C}_1} (\boldsymbol{x}_n-{\pmb \mu}_1)(\boldsymbol{x}_n-{\pmb \mu}_1)^T\\ {\boldsymbol S}_2 &\ = \frac{1}{N_2}\sum_{n \in \mathcal{C}_2} (\boldsymbol{x}_n-{\pmb \mu}_2)(\boldsymbol{x}_n-{\pmb \mu}_2)^T \end{align}$$  

가우시안 분포의 최대우도를 구하는 방법을 그대로 쓰면 결국은  

$$\Sigma = {\boldsymbol S}$$  

## 복습 - 가우시안 분포의 최대우도 (Maximum Likelihood for the Gaussian)
> <https://marquis08.github.io/devcourse2/probabilitydistributions/mathjax/ML-basics-Probability-Distributions-2/#ml---%EA%B3%B5%EB%B6%84%EC%82%B0>  

다음으로 우도를 최대화하는 공분산행렬 $$\Sigma_{ML}$$을 찾아보자.  

$$l(\Lambda) = \frac{N}{2}\ln\vert \Lambda\vert - \frac{1}{2}\sum_{n=1}^{N}tr((\boldsymbol{x}_{n}-\boldsymbol{\mu})(\boldsymbol{x}_{n}-\boldsymbol{\mu})^{T}\Lambda) = \frac{N}{2}\ln\vert \Lambda\vert - \frac{1}{2}tr(\boldsymbol{S}\Lambda)$$  

$$\boldsymbol{S} = \sum_{i=1}^{N}(\boldsymbol{x}_{n}-\boldsymbol{\mu})(\boldsymbol{x}_{n}-\boldsymbol{\mu})^{T}$$  


$$\frac{\partial l(\Lambda)}{\partial \Lambda} = \frac{N}{2}(\Lambda^{-1})^{T}-\frac{1}{2}\boldsymbol{S}^{T} = 0$$  

$$(\Lambda_{ML}^{-1}) = \Sigma_{ML} = \frac{1}{N}\boldsymbol{S}$$  

$$\Sigma_{ML} = \frac{1}{N}\sum_{i=1}^{N}(\boldsymbol{x}_{n}-\boldsymbol{\mu})(\boldsymbol{x}_{n}-\boldsymbol{\mu})^{T}$$  

위의 식 유도를 위해 아래의 기본적인 선형대수 결과를 사용하였다.  

- \\(\vert \boldsymbol{A}^{-1}\vert = l/\vert \boldsymbol{A}\vert\\)
- \\(\boldsymbol{x}^{T}\boldsymbol{Ax} = tr(\boldsymbol{x}^{T}\boldsymbol{Ax}) = tr(\boldsymbol{xx}^{T}\boldsymbol{A}) \\)
- \\(tr(\boldsymbol{A}) + tr(\boldsymbol{B}) = tr(\boldsymbol{A}+\boldsymbol{B}) \\)
- \\(\frac{\partial}{\partial\boldsymbol{A}}tr(\boldsymbol{BA}) = \boldsymbol{B}^{T} \\)
- \\(\frac{\partial}{\partial\boldsymbol{A}}\ln\vert \boldsymbol{A}\vert = (\boldsymbol{A}^{-1})^{T}\\)

## 입력이 이산값일 경우 (Discrete features)

각 특성 $$x_i$$가 0과 1중 하나의 값만을 가질 수 있는 경우

클래스가 주어졌을 때 특성들이 조건부독립(conditional independence)이라는 가정을 할 경우 문제는 단순화된다. 이것을 naive Bayes가정이라고 한다. 이 때 $$p(\boldsymbol{x}\vert \mathcal{C}_k)$$는 다음과 같이 분해된다.  

$$p(\boldsymbol{x}\vert \mathcal{C}_k) = \prod_{i=1}^{D}\mu_{ki}^{x_i}(1\mu_{ki})^{1-x_i}$$  

따라서,  

$$a_k(\boldsymbol{x})=\ln p(\boldsymbol{x}\vert \mathcal{C}_k)p(\mathcal{C}_k)$$  

$$a_k(\boldsymbol{x})=\sum_{i=1}^{D}\left\{x_i\ln \mu_{ki}+(1-x_i)\ln(1 - \mu_{ki})\right\}+\ln p(\mathcal{C}_k)$$  

# 확률적 식별 모델 (Probabilistic Discriminative Models)

앞의 생성모델에서는 $$p(\mathcal{C}_k\vert \boldsymbol{x})$$를 $$\boldsymbol{x}$$의 선형함수가 logistic sigmoid 또는 softmax를 통과하는 식으로 표현되는 것을 보았다. 즉, K=2인 경우  

$$p(\mathcal{C}_1\vert \boldsymbol{x}) = \sigma(\boldsymbol{w}^T\boldsymbol{x}+w_0)$$  

그리고 파라미터들 $$\boldsymbol{w}$$와 $$w_0$$를 구하기 위해서 확률분포들 $$p(\boldsymbol{x}\vert \mathcal{C}_k)$$, $$p(\mathcal{C}_k)$$의 파라미터들을 MLE로 구했다.  

대안적인 방법은 $$p(\mathcal{C}_k\vert \boldsymbol{x})$$를 $$\boldsymbol{x}$$에 관한 함수로 파라미터화 시키고 이 파라미터들을 직접 MLE를 통해 구하는 것이다.  

이제부터는 입력벡터 $$\boldsymbol{x}$$대신 비선형 기저함수(basis function)들 $$\phi(\boldsymbol{x})$$를 사용할 것이다.  

## 로지스틱 회귀 (Logistic regression)

클래스 $$\mathcal{C}_1$$의 사후확률은 특성벡터 $$\phi$$의 선형함수가 logistic sigmoid를 통과하는 함수로 아래와 같이 표현된다.  

$$p(\mathcal{C}_1\vert \phi)=y(\phi)=\sigma(\boldsymbol{w}^T\phi)$$  

$$\sigma(a)=\frac{1}{1+\exp(-a)}$$  

$$p(\mathcal{C}_2\vert \phi) = 1 - p(\mathcal{C}_1\vert \phi)$$  

$$\phi$$가 $$M$$ 차원이라면 구해야 할 파라미터($$\boldsymbol{w}$$)의 개수는 $$M$$개이다. 생성모델에서는 $$M(M+5)/2+1$$개의 파라미터를 구해야 한다.  

### 최대우도해

\- 데이터셋: $$\{\phi_n, t_n\}$$, $$n=1,\ldots,N$$  
\- $$t_n \in \{0, 1\}$$  
\- $$\boldsymbol{t} = (t_1,\ldots,t_N)^T$$  
\- $$\phi_n = \phi(\boldsymbol{x}_n)$$  
\- $$y_n = p(\mathcal{C}_1\vert \phi_n)$$  

우도함수는  

$$p(\boldsymbol {T}\vert \boldsymbol{w}) = \prod_{n=1}^{N}y_n^{t_n}(1-y_n)^{1-t_n}$$

음의 로그 우도 (the negative logarithm of the likelihood)  

$$E(\boldsymbol{w})= - \ln{p(\boldsymbol {T}\vert \boldsymbol{w})} = - \sum_{n=1}^{N}\left\{t_n\ln{y_n}+(1-t_n)\ln(1-y_n)\right\}$$  

$$y_n = \sigma(a_n)$$, $$a_n = \boldsymbol{w}^T\phi_n$$  

이것을 크로스 엔트로피 에러함수(cross entropy error function)라고 부른다.

Cross entropy의 일반적인 정의  

$$H(p,q) = - \mathbb{E}_p[\ln q]$$  

이산확률변수의 경우   

$$H(p,q) = - \sum_{x}p(x)\ln q(x)$$  

일반적으로 Cross entropy가 최소화될 때 두 확률분포의 차이가 최소화된다. 따라서 에러함수 $$E(\boldsymbol{w})$$를 최소화시키는 것을  

\- 우도를 최대화시키는 것  
\- 모델의 예측값(의 분포)과 목표변수(의 분포)의 차이를 최소화시키는 것

두 가지의 관점에서 이해할 수 있다.  

에러함수의 $$\boldsymbol{w}$$에 대한 gradient를 구해보자.  

$$E_n(\boldsymbol{w})= - \left\{t_n\ln{y_n}+(1-t_n)\ln(1-y_n)\right\}$$  

라고 정의하면  

$$\triangledown E(\boldsymbol{w}) = \sum_{n=1}^N \triangledown E_n(\boldsymbol{w})$$  

$$\begin{align} \triangledown E_n(\boldsymbol{w}) &\ = \frac{\partial E_n(\boldsymbol{w})}{\partial y_n}\frac{\partial y_n}{\partial a_n}\triangledown a_n\\ &\ = \left\{ \frac{1-t_n}{1-y_n} - \frac{t_n}{y_n}\right\} y_n(1-y_n)\phi_n\\ &\ = (y_n - t_n)\phi_n \end{align}$$  

따라서  

$$\triangledown E(\boldsymbol{w}) = \sum_{n=1}^N (y_n - t_n)\phi_n$$  

## 다중클래스 로지스틱 회귀 (Multiclass logistic regression)

$$p(\mathcal{C}_k\vert \phi) = y_k(\phi) = \frac{\exp(a_k)}{\sum_j \exp(a_j)}$$  

$$a_k = \boldsymbol{w}_k^T \phi$$  

### 우도함수

특성벡터 $$\phi_n$$를 위한 목표벡터 $$\boldsymbol{t}_n$$는 클래스에 해당하는 하나의 원소만 1이고 나머지는 0인 1-of-K 인코딩 방법으로 표현된다.   

$$p(\boldsymbol {T}\vert \boldsymbol{w}_1,...\boldsymbol{w}_K) = \prod_{n=1}^{N}\prod_{k=1}^{K} p(\mathcal{C}_k\vert \phi_n)^{t_{nk}} = \prod_{n=1}^{N}\prod_{k=1}^{K}y_{nk}^{t_{nk}}$$  

$$y_{nk} = y_k(\phi_n)$$, $$\boldsymbol {T}$$는 $$t_{nk}$$를 원소로 가지고 있는 크기가 $$N \times K$$인 행렬  

음의 로그 우도  

$$E(\boldsymbol{w}_1, ..., \boldsymbol{w}_K) = - \ln p(\boldsymbol {T}\vert \boldsymbol{w}_1, ...,\boldsymbol{w}_K) = - \sum_{n=1}^{N} \sum_{k=1}^{K} t_{nk}\ln(y_{nk})$$  

$$\boldsymbol{w}_j$$에 대한 gradient를 구한다. 먼저 하나의 샘플 $$\phi_n$$에 대한 에러  

$$E_n(\boldsymbol{w}_1,\ldots,\boldsymbol{w}_K) = - \sum_{k=1}^{K} t_{nk}\ln(y_{nk})$$  

를 정의하면  

$$\nabla_{ \boldsymbol{w}_j }E(\boldsymbol{w}_1, ...,\boldsymbol{w}_K) = \sum_{n=1}^{N}\nabla_{ \boldsymbol{w}_j }E_n(\boldsymbol{w}_1, ...,\boldsymbol{w}_K)$$

다음 함수들 사이의 관계를 주목하라.  

\- $$E_n$$와 $$\boldsymbol{w}_j$$의 관계는 오직 $$a_{nj}$$에만 의존한다($$a_{nk}, k\neq j$$는 $$\boldsymbol{w}_j$$의 함수가 아니다).  
\- $$E_n$$은 $$y_{n1},\ldots,y_{nK}$$의 함수이다.  
\- $$y_{nk}$$는 $$a_{n1},\ldots,a_{nK}$$의 함수이다.

$$\begin{align} \nabla_{ \boldsymbol{w}_j }E_n &\ = \frac{\partial E_n}{\partial a_{nj}} \frac{\partial a_{nj}}{\partial \boldsymbol{w}_j}\\ &\ = \frac{\partial E_n}{\partial a_{nj}}\phi_n\\ &\ = \sum_{k=1}^K \left( \frac{\partial E_n}{\partial y_{nk}} \frac{\partial y_{nk}}{\partial a_{nj}} \right)\phi_n\\ &\ = \phi_n \sum_{k=1}^K \left\{ - \frac{t_{nk}}{y_{nk}}y_{nk}(I_{kj}-y_{nj}) \right\}\\ &\ = \phi_n \sum_{k=1}^K t_{nk}(y_{nj} - I_{kj})\\ &\ = \phi_n \left( y_{nj}\sum_{k=1}^K t_{nk} - \sum_{k=1}^K t_{nk}I_{kj} \right)\\ &\ = \phi_n (y_{nj} - t_{nj}) \end{align}$$  

따라서  

$$\nabla_{ \boldsymbol{w}_j }E(\boldsymbol{w}_1, ...,\boldsymbol{w}_K) = \sum_{n=1}^{N} (y_{nj}-t_{nj})\phi_n$$  

# 실습
## Gradient Descent (batch)
```python
In [ ]:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns

In [ ]:

X, t = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, random_state=14)

t = t[:,np.newaxis]

sns.set_style('white')
sns.scatterplot(X[:,0],X[:,1],hue=t.reshape(-1));

In [ ]:

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

In [ ]:

def compute_cost(X, t, w):
    N = len(t)
    h = sigmoid(X @ w)
    epsilon = 1e-5
    cost = (1/N)*(((-t).T @ np.log(h + epsilon))-((1-t).T @ np.log(1-h + epsilon)))
    return cost

In [ ]:

def gradient_descent(X, t, w, learning_rate, iterations):
    N = len(t)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        w = w - (learning_rate/N) * (X.T @ (sigmoid(X @ w) - t))
        cost_history[i] = compute_cost(X, t, w)

    return (cost_history, w)

In [ ]:

def predict(X, w):
    return np.round(sigmoid(X @ w))

In [ ]:

N = len(t)

X = np.hstack((np.ones((N,1)),X))
M = np.size(X,1)
w = np.zeros((M,1))

iterations = 1000
learning_rate = 0.01

initial_cost = compute_cost(X, t, w)

print("Initial Cost is: {} \n".format(initial_cost))

(cost_history, w_optimal) = gradient_descent(X, t, w, learning_rate, iterations)

print("Optimal Parameters are: \n", w_optimal, "\n")

plt.figure()
sns.set_style('white')
plt.plot(range(len(cost_history)), cost_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()

In [ ]:

## Accuracy

y_pred = predict(X, w_optimal)
score = float(sum(y_pred == t))/ float(len(t))

print(score)

In [ ]:

slope = -(w_optimal[1] / w_optimal[2])
intercept = -(w[0] / w_optimal[2])

sns.set_style('white')
sns.scatterplot(X[:,1],X[:,2],hue=t.reshape(-1));

ax = plt.gca()
ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = intercept + (slope * x_vals)
plt.plot(x_vals, y_vals, c="k");

Stochastic Gradient Descent
In [ ]:

def sgd(X, t, w, learning_rate, iterations):
    N = len(t)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        i = i % N
        w = w - learning_rate * (X[i, np.newaxis].T * (sigmoid(X[i] @ w) - t[i]))
        cost_history[i] = compute_cost(X[i], t[i], w)

    return (cost_history, w)

In [ ]:

X, t = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, random_state=14)

t = t[:,np.newaxis]

N = len(t)

X = np.hstack((np.ones((N,1)),X))
M = np.size(X,1)
w = np.zeros((M,1))

iterations = 2000
learning_rate = 0.01

initial_cost = compute_cost(X, t, w)

print("Initial Cost is: {} \n".format(initial_cost))

(cost_history, w_optimal) = sgd(X, t, w, learning_rate, iterations)

print("Optimal Parameters are: \n", w_optimal, "\n")

plt.figure()
sns.set_style('white')
plt.plot(range(len(cost_history)), cost_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()

In [ ]:

## Accuracy

y_pred = predict(X, w_optimal)
score = float(sum(y_pred == t))/ float(len(t))

print(score)

Mini-batch Gradient Descent
In [ ]:

def batch_gd(X, t, w, learning_rate, iterations, batch_size):
    N = len(t)
    cost_history = np.zeros((iterations,1))
    shuffled_indices = np.random.permutation(N)
    X_shuffled = X[shuffled_indices]
    t_shuffled = t[shuffled_indices]

    for i in range(iterations):
        i = i % N
        X_batch = X_shuffled[i:i+batch_size]
        t_batch = t_shuffled[i:i+batch_size]
        # batch가 epoch 경계를 넘어가는 경우, 앞 부분으로 채워줌
        if X_batch.shape[0] < batch_size:
            X_batch = np.vstack((X_batch, X_shuffled[:(batch_size - X_batch.shape[0])]))
            t_batch = np.vstack((t_batch, t_shuffled[:(batch_size - t_batch.shape[0])]))
        w = w - (learning_rate/batch_size) * (X_batch.T @ (sigmoid(X_batch @ w) - t_batch))
        cost_history[i] = compute_cost(X_batch, t_batch, w)

    return (cost_history, w)

In [ ]:

X, t = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, random_state=14)

t = t[:,np.newaxis]

N = len(t)

X = np.hstack((np.ones((N,1)),X))
M = np.size(X,1)
w = np.zeros((M,1))

iterations = 1000
learning_rate = 0.01

initial_cost = compute_cost(X, t, w)

print("Initial Cost is: {} \n".format(initial_cost))

(cost_history, w_optimal) = batch_gd(X, t, w, learning_rate, iterations, 32)

print("Optimal Parameters are: \n", w_optimal, "\n")

plt.figure()
sns.set_style('white')
plt.plot(range(len(cost_history)), cost_history, 'r')
plt.title("Convergence Graph of Cost Function")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.show()

In [ ]:

## Accuracy

y_pred = predict(X, w_optimal)
score = float(sum(y_pred == t))/ float(len(t))

print(score)
```


# Appendix

## MathJax

left align:  

$$\begin{align} &\ A = AAAA \\ &\ AAAAA = A \\ &\  AAA = AA \end{align}$$  

```
$$\begin{align} &\ A = AAAA \\ &\ A = A \\ &\  A = AA \end{align}$$
```

escape unordered list  
\- $$cov[\boldsymbol{x}_{a}] = \Sigma_{aa}$$:  
```
\- $$cov[\boldsymbol{x}_{a}] = \Sigma_{aa}$$ 
```

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

Matrix with parenthesis $$\pmatrix{a_{11} & a_{12} & \ldots & a_{1n} \cr a_{21} & a_{22} & \ldots & a_{2n} \cr \vdots & \vdots & \ddots & \vdots \cr a_{m1} & a_{m2} & \ldots & a_{mn} }$$:  
```
$$\pmatrix{a_{11} & a_{12} & \ldots & a_{1n} \cr a_{21} & a_{22} & \ldots & a_{2n} \cr \vdots & \vdots & \ddots & \vdots \cr a_{m1} & a_{m2} & \ldots & a_{mn} }$$
```
## References

> Pattern Recognition and Machine Learning: <https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf>  
