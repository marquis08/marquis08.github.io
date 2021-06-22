---
date: 2021-06-23 02:41
title: "ML basics - Linear Models for Regression"
categories: DevCourse2 Regression MathJax
tags: DevCourse2 Regression MathJax
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---


# 선형 기저 함수 모델
가장 단순한 형태의 선형모델  

$$y(\boldsymbol{x}, \boldsymbol{w}) = w_{0}+w_{1}x_{1}+\cdots w_{D}x_{D}$$  

$$\boldsymbol{x} = (x_{1},\cdots,x_{D})^{T}$$  

이 모델의 파라미터는 $$\boldsymbol{w} = (w_{0},\cdots,w_{D})^{T}$$ 벡터이다. 위 함수는 파라미터 $$\boldsymbol{w}$$에 대해 선형일 뿐만 아니라 입력 데이터 $$\boldsymbol{x}$$에 대해서도 선형이다.  

$$\boldsymbol{x}$$에 대해 비선형인 함수를 만들고 싶다면?  

$$y(\boldsymbol{x}, \boldsymbol{w}) = w_{0}+\sum_{j=1}^{M-1}w_{j}\phi_{j}(\boldsymbol{x})$$  

$$y(\boldsymbol{x}, \boldsymbol{w}) = \sum_{j=0}^{M-1}w_{j}\phi_{j}(\boldsymbol{x}) = \boldsymbol{w}^{T}\phi(\boldsymbol{x})$$  

$$\phi_{0}(\boldsymbol{x}) = 1$$

$$\boldsymbol{x}$$ 에 대해 비선형인 함수 $$\phi_{j}(\boldsymbol{x})$$ 를 기저함수(basis function)라고 부른다.  

앞에서 몇 가지 기저함수를 이미 사용한 적이 있다.  

- 다항식 기저함수  

$$\phi_{j}(x) = x^{j}$$  

- 가우시안 기저함수  

$$\phi_{j}(x) = \exp\left\{-\frac{(x-\mu_{j})^2}{2s^{2}}  \right\}$$  

- 시그모이드 기저함수  

$$\phi_{j}(x) = \sigma\left(\frac{x-\mu_{j}}{s}  \right)$$  

$$\sigma(a) = \frac{1}{1+\exp(-a)}$$  

그림

# 최대우도와 최소제곱법 (Maximum Likelihood and Least Squares)
에러함수가 가우시안 노이즈를 가정할 때 최대우도로부터 유도될 수 있다는 것을 살펴본 적이 있다. 조금 더 자세히 살펴보자.  

$$t = y(\boldsymbol{x}, \boldsymbol{w}) + \epsilon$$  

- $$y(\boldsymbol{x}, \boldsymbol{w})$$ 는 결정론적 함수(deterministic)
- $$\epsilon$$는 가우시안 분포 $$\mathcal{N}(\epsilon\vert 0, \beta^{-1})$$를 따르는 노이즈 확률변수  

따라서 $$t$$ 의 분포는 다음과 같다.  

$$p(t\vert \boldsymbol{x}, \boldsymbol{w}, \beta) = \mathcal{N}(t\vert y(\boldsymbol{x}, \boldsymbol{w}), \beta^{-1})$$  

제곱합이 손실함수로 쓰이는 경우(squared loss function), 새로운 $$\boldsymbol{x}$$ 가 주어졌을 때 $$t$$ 의 최적의 예측값(optimal prediction)은 $$t$$ 의 조건부 기댓값이다.  
$$t$$ 가 위의 분포를 따르는 경우 조건부 기댓값은 다음과 같다.  

$$\mathbb{E}[t\vert \boldsymbol{x}] = \int tp(t\vert \boldsymbol{x})dt = y(\boldsymbol{x}, \boldsymbol{w})$$  

이제 파라미터인 $$\boldsymbol{w}$$ 를 찾기 위해 최대우도 추정법을 사용해보자.  

- 입력값 $$\boldsymbol{X} = \boldsymbol{x}_{1},\cdots,\boldsymbol{x}_{N}$$  
- 출력값은 $$\boldsymbol{t} = \boldsymbol{t}_{1},\cdots,\boldsymbol{t}_{N}$$  

우도함수는  

$$p(\boldsymbol{t}\vert \boldsymbol{X}, \boldsymbol{w}, \beta) = \prod_{n=1}^{N}\mathcal{N}(t_{n}\vert \boldsymbol{w}^{T}\phi(\boldsymbol{x}_{n}), \beta^{-1})$$  

로그 우도함수는  

$$\ln p(\boldsymbol{t}\vert \boldsymbol{w}, \beta) = \sum_{n=1}^{N}\ln \mathcal{N}(t_{n}\vert \boldsymbol{w}^{T}\phi(\boldsymbol{x}_{n}), \beta^{-1}) = \frac{N}{2}\ln \beta - \frac{N}{2}\ln(2\pi) - \beta\boldsymbol{E}_{D}(\boldsymbol{w})$$  

$$\boldsymbol{E}_{D}(\boldsymbol{w}) = \frac{1}{2}\sum_{n=1}^{N}\{ t_{n}- \boldsymbol{w}^{T}\phi(\boldsymbol{x}_{n})  \}^{2}$$  

따라서, 로그 우도함수를 최대화시키는 $$\boldsymbol{w}$$ 값은 $$\boldsymbol{E}_{D}(\boldsymbol{w})$$ 로 주어진 제곱합 에러함수를 최소화시키는 값과 동일하다는 것을 알 수 있음.  

$$\boldsymbol{w}$$ 에 대한 기울기벡터(gradient vector)는  

$$\nabla\ln p(\boldsymbol{t}\vert \boldsymbol{w}, \beta) = \sum_{n=1}^{N}\{ \boldsymbol{t}_{n} - \boldsymbol{w}^{T}\phi(\boldsymbol{x}_{n}) \}\phi(\boldsymbol{x}_{n})^{T}$$  

따라서 $$\boldsymbol{w}$$ 의 최적값은  

$$\boldsymbol{w}_{ML} = (\Phi^{T}\Phi)^{-1}\Phi^{T}\boldsymbol{t}$$  

위 식을 normal equations라고 부른다.  

$$\Phi = \begin{pmatrix} \phi_{0}(\boldsymbol{x}_{1}) & \phi_{1}(\boldsymbol{x}_{1}) & \ldots & \phi_{M-1}(\boldsymbol{x}_{1}) \cr \phi_{0}(\boldsymbol{x}_{2}) & \phi_{1}(\boldsymbol{x}_{2}) &\ldots & \phi_{M-1}(\boldsymbol{x}_{2}) \cr \vdots & \vdots & \ddots & \vdots \cr \phi_{0}(\boldsymbol{x}_{N}) & \phi_{1}(\boldsymbol{x}_{N}) & \ldots & \phi_{M-1}(\boldsymbol{x}_{N}) \end{pmatrix}$$  

Moore-Penrose pseudo-inverse  

$$\Phi^{\dagger} \equiv (\Phi^{T}\Phi)^{-1}\Phi^{T}$$  

Normal Equation 유도하기  

편향 파라미터 (bias parameter) $$\boldsymbol{w}_{0}$$  

$$\boldsymbol{E}_{D}(\boldsymbol{w}) = \frac{1}{2}\sum_{n=1}^{N}\{ \boldsymbol{t}_{n} - \boldsymbol{w}_{0}-\sum_{j=1}^{M-1}\boldsymbol{w}_{j}\phi_{j}(\boldsymbol{x}_{n}) \}^{2}$$  

$$\boldsymbol{w}_{0} = \bar t - \sum_{j=1}^{M-1}\boldsymbol{w}_{j}\bar{\phi_{j}}$$  

$$\bar t = \frac{1}{N}\sum_{n=1}^{N}t_{n},\  \bar{\phi_{j}} = \frac{1}{N}\sum_{n=1}^{N}\phi(\boldsymbol{x}_{n})$$  

$$\beta$$ 의 최적값은  

$$\frac{1}{\beta_{ML}} = \frac{1}{N}\sum_{n=1}^{N}{\boldsymbol{t}_{n} - \boldsymbol{w}_{ML}^{T}\phi(\boldsymbol{x}_{n})}^{2}$$  

- 벡터의 집합 ($$\{ \boldsymbol{x}_{1},\cdots,\boldsymbol{x}_{n} \}$$) 에 대한 생성(span)  

$$span(\{ \boldsymbol{x}_{1},\cdots,\boldsymbol{x}_{n} \}) = \left\{ \boldsymbol{v}:\boldsymbol{v} = \sum_{i=1}^{n}\alpha_{i}\boldsymbol{x}_{i}, \alpha_{i} \in \mathbb{R} \right\}$$  

- 행렬의 치역(range)  
행렬 $$\boldsymbol{A} \in \mathbb{R}^{m\times n}$$ 의 치역 $$\mathcal{R}(\boldsymbol{A})$$ 는 $$\boldsymbol{A}$$ 의 모든 열들에 대한 생성(span)이다.  

$$\mathcal{R}(\boldsymbol{A}) = \left\{ \boldsymbol{v}\in \mathcal{R}^{m}:\boldsymbol{v} = \boldsymbol{Ax},\ \boldsymbol{x}\in \mathcal{R}^{n}  \right\}$$  

- 벡터의 사영(projection)  

벡터 $$\boldsymbol{t} \in \mathbb{R}^{m}$$ 의 $$span(\{ \boldsymbol{x}_{1},\cdots,\boldsymbol{x}_{n} \})(\boldsymbol{x}_{i}\in \mathcal{R}^{m})$$ 으로의 사영은 $$span(\{ \boldsymbol{x}_{1},\cdots,\boldsymbol{x}_{n} \})$$ 에 속한 벡터 중 $$\boldsymbol{t}$$  에 가장 가까운 벡터로 정의된다.  

$$Proj(\boldsymbol{t}; \{ \boldsymbol{x}_{1},\cdots,\boldsymbol{x}_{n} \}) = \arg\min\ _{v\in span(\{ \boldsymbol{x}_{1},\cdots,\boldsymbol{x}_{n} \})}\Vert\boldsymbol{t}-\boldsymbol{v} \Vert_{2}$$  

$$Proj(\boldsymbol{t};\boldsymbol{A})$$ 은 행렬 $$\boldsymbol{A}$$  의 치역으로의 사영이다. $$\boldsymbol{A}$$ 의 열들이 선형독립이면,  

$$Proj(\boldsymbol{t};\boldsymbol{A}) =  \arg\min\ _{v\in \mathcal{R}(\boldsymbol{A})}\Vert\boldsymbol{t}-\boldsymbol{v} \Vert_{2} = \boldsymbol{A}(\boldsymbol{A}^{T}\boldsymbol{A})^{-1}\boldsymbol{A}^{T}\boldsymbol{t}$$  

![그림 여기에](){}  



# 온라인 학습 (Sequantial Learning)
배치학습 vs. 온라인 학습

Stochastic gradient decent  

에러함수가 $$\boldsymbol{E} = \sum_{n}\boldsymbol{E}_{n}$$ 이라고 하자.  

$$\boldsymbol{w}^{\tau+1}=\boldsymbol{w}^{\tau} - \eta\nabla\boldsymbol{E}_{n}$$  

제곱합 에러함수인 경우  

$$\boldsymbol{w}^{\tau+1}=\boldsymbol{w}^{\tau} + \eta(t_{n}-\boldsymbol{w}^{\tau}T\phi_{n})\phi_{n}$$  

$$\phi_{n} = \phi(\boldsymbol{x}_{n})$$  

# 실습 (대규모의 선형회귀)

# 규제화된 최소제곱법 (Regularized Least Squares)
$$\boldsymbol{E}_{D}(\boldsymbol{w}) + \lambda\boldsymbol{E}_{w}(\boldsymbol{w})$$  

가장 단순한 형태는  

$$\boldsymbol{E}_{w}(\boldsymbol{w}) = \frac{1}{2}\boldsymbol{w}^{T}\boldsymbol{w}$$  

$$\boldsymbol{E}_{D}(\boldsymbol{w}) = \frac{1}{2}\sum_{n=1}^{N}\{ t_{n} - \boldsymbol{w}^{T}\phi(\boldsymbol{x}_{n}) \}^2$$  

최종적인 에러함수는  

$$\frac{1}{2}\sum_{n=1}^{N}\{ t_{n} - \boldsymbol{w}^{T}\phi(\boldsymbol{x}_{n}) \}^2 + \frac{\lambda}{2}\boldsymbol{w}^{T}\boldsymbol{w}$$  

$$\boldsymbol{w}$$ 의 최적값은  

$$\boldsymbol{w}^{T} = (\lambda\boldsymbol{I}+\phi^{T}\phi)^{-1}\phi^{T}\boldsymbol{t}$$  

일반화된 규제화  

$$\boldsymbol{E}(\boldsymbol{w}) = \frac{1}{2}\sum_{n=1}^{N}\{ t_{n} - \boldsymbol{w}^{T}\phi(\boldsymbol{x}_{n}) \}^2 + \frac{1}{2}\sum_{j=1}^{M}\vert\boldsymbol{w}_{j} \vert^{q}$$  

Lasso 모델($$q=1$$)  

- Constrained minimization 문제로 나타낼 수 있다.  

$$\sum_{j=1}^{M}\vert\boldsymbol{w}_{j} \vert^{q} \leq\eta$$  

![그림](){: .align-center}  

# 편향-분산 분해(Bias-Variance Decomposition)
모델이 과적합되는 현상에 대한 이론적인 분석  

제곱합 손실함수가 주어졌을 때의 최적 예측값  

$$h(\boldsymbol{x}) = \mathbb{E}[t\vert \boldsymbol{x}] = \int tp(t\vert \boldsymbol{x})dt$$  

손실함수의 기댓값  

$$\mathbb{E}[\boldsymbol{L}] = \int \{y(\boldsymbol{x}) - h(\boldsymbol{x}) \}^2p(\boldsymbol{x})d\boldsymbol{x} + \int\int \{ h(\boldsymbol{x}-t) \}^2p(\boldsymbol{x},t)d\boldsymbol{x}dt$$  

제한된 데이터셋 $$D$$ 만 주어져 있기 때문에 $$h(\boldsymbol{x})$$ 를 정확히 알 수 는 없다. 대신 파라미터화 된 함수 $$y(\boldsymbol{x}, \boldsymbol{w})$$  를 사용해 최대한 손실함수의 기댓값을 최소화하고자 한다.  

제한된 데이터로 인해 발생하는 모델의 불확실성을 어떻게든 표현해야 한다.  

- 베이지안 방법: 모델 파라미터 $$\boldsymbol{w}$$  의 사후확률분포를 계산한다.  
- 빈도주의 방법: 모델 파라미터 $$\boldsymbol{w}$$  의 점추정 값을 구하고 여러 개의 데이터셋을 가정했을 때 발생하는 평균적인 손실을 계산하는 '가상의 실험'을 통해 점추정 값의 불확실성을 해석한다.  

특정 데이터셋 $$D$$ 에 대한 손실을  

$$\boldsymbol{L}(\mathcal{D}) = \{y(\boldsymbol{x};\mathcal{D})-h(\boldsymbol{x}) \}^2$$  

라고 하자. 손실함수의 기댓값은  

$$\mathbb{E}[\boldsymbol{L}(\mathcal{D})] = \int \{y(\boldsymbol{x};\mathcal{D})-h(\boldsymbol{x})\}^2 p(\boldsymbol{x})d\boldsymbol{x} + noise$$  

여러 개의 데이터셋 $$\mathcal{D}_{1},\cdots,\mathcal{D}_{L}$$ 이 주어졌을 때 이 값들의 평균을 생각해보자.  

$$\frac{1}{\boldsymbol{L}}\sum_{l=1}^{L} \left[ \int \{y(\boldsymbol{x};\mathcal{D}^{(i)})-h(\boldsymbol{x})\}^2 p(\boldsymbol{x})d\boldsymbol{x} + noise \right] = \int \mathbb{E}_{\mathcal{D}} \left[ \{y(\boldsymbol{x};\mathcal{D})-h(\boldsymbol{x}) \}^2 \right] p(\boldsymbol{x})d\boldsymbol{x} + noise$$  


$$\{y(\boldsymbol{x};\mathcal{D}) -\boldsymbol{E}_{\mathcal{D}}\left[y(\boldsymbol{x};\mathcal{D})\right] +\boldsymbol{E}_{\mathcal{D}}\left[y(\boldsymbol{x};\mathcal{D})\right] -h(\boldsymbol{x}) \}^2$$  
$$= \{y(\boldsymbol{x};\mathcal{D}) -\boldsymbol{E}_{\mathcal{D}}\left[y(\boldsymbol{x};\mathcal{D})\right]\}^2 + \{\boldsymbol{E}_{\mathcal{D}}\left[y(\boldsymbol{x};\mathcal{D})\right] - h(\boldsymbol{x})\}^2 + 2\{y(\boldsymbol{x};\mathcal{D}) -\boldsymbol{E}_{\mathcal{D}}\left[y(\boldsymbol{x};\mathcal{D})\right]\}\{\boldsymbol{E}_{\mathcal{D}}\left[y(\boldsymbol{x};\mathcal{D})\right] - h(\boldsymbol{x})\}$$  


따라서  

$$\mathbb{E}_{\mathcal{D}} \left[ \{y(\boldsymbol{x};\mathcal{D})-h(\boldsymbol{x}) \}^2 \right] = \{\mathbb{E}_{\mathcal{D}} \left[y(\boldsymbol{x};\mathcal{D})\right]-h(\boldsymbol{x})\}^2 + \mathbb{E}_{\mathcal{D}}\left[ \{y(\boldsymbol{x};\mathcal{D}) - \mathbb{E}_{\mathcal{D}}\left[  y(\boldsymbol{x};\mathcal{D}) \right]\}^2 \right]$$  

정리하자면  

$$\text{Expected loss} = (\text{bias})^2 + \text{variance} + \text{noise}$$  

$$(\text{bias})^2 = \int \{\mathbb{E}_{\mathcal{D}}\left[  y(\boldsymbol{x};\mathcal{D}) \right] - h(\boldsymbol{x}) \}^2 p(\boldsymbol{x})d\boldsymbol{x}$$  

$$\text{variance} = \int \mathbb{E}_{\mathcal{D}} \left[\{  y(\boldsymbol{x};\mathcal{D}) - \mathbb{E}_{\mathcal{D}}\left[ y(\boldsymbol{x};\mathcal{D}) \right] \}^2\right] p(\boldsymbol{x})d\boldsymbol{x}$$  

$$\text{noise} = \int\int \{h(\boldsymbol{x})-t\}^2 p(\boldsymbol{x},t)d\boldsymbol{x}$$  

예제  
$$h(\boldsymbol{x}) = \sin(2\pi x)$$  

$$\begin{align}\bar y(x) &= \frac{1}{L}\sum_{l=1}^{L}y^{(l)}(x) \\\\ (\text{bias})^2 &= \frac{1}{N}\sum_{n=1}^{N}\{ \bar y(x_{n})-h(x_{n})  \}^2 \\\\ \text{variance} &= \frac{1}{N}\sum_{n=1}^{N}\frac{1}{L}\sum_{l=1}^{L}\{ y^{(l)}(x_{n})-\bar y(x_{n})  \}^2 \end{align}$$  

그림  

- 별개의 테스트 데이터셋에 대한 결과  

그림  

# 베이지안 선형회귀 (Bayesian Linear Regression)
- 파라미터 $$\boldsymbol{w}$$ 의 사전확률을 다음과 같은 가우시안 분포라고 하자.  

$$p(\boldsymbol{w}) = \mathcal{N}(\boldsymbol{w}\vert \boldsymbol{m}_{0}, \boldsymbol{S}_{0})$$  

- 우도  

$$\begin{align} p(\boldsymbol{t}\vert \boldsymbol{w}) &= p(t_{1},\cdots,t_{N}\vert \boldsymbol{w}) \\ &= \prod_{n=1}^{N}\mathcal{N}(t_{n}\vert \boldsymbol{w}^{T}\phi(\boldsymbol{x}_{n}), \beta^{-1}) \\ &= \mathcal{N}(\boldsymbol{t}\vert \Phi\boldsymbol{w}, \beta^{-1}\boldsymbol{I}) \end{align}$$  

- 복습 (가우시안 분포를 위한 베이즈 정리)  

$$p(\boldsymbol{x})$$ 와 $$p(\boldsymbol{y}\vert \boldsymbol{x})$$  가 다음과 같이 주어진다고 하자.  

$$\begin{align} p(\boldsymbol{x}) &= \mathcal{N}(\boldsymbol{x}\vert \boldsymbol{\mu}, \Lambda^{-1})\\ p(\boldsymbol{y}\vert \boldsymbol{x}) &= \mathcal{N}(\boldsymbol{y}\vert \boldsymbol{Ax} + \boldsymbol{b}, \boldsymbol{L}^{-1}) \end{align}$$  

조건부 확률 $$p(\boldsymbol{y}\vert \boldsymbol{x})$$  의 평균과 공분산은 다음과 같다.  

$$\begin{align}\mathbb{E}[\boldsymbol{x}\vert \boldsymbol{y}] &= (\Lambda + \boldsymbol{A}^{T}\boldsymbol{LA} )^{-1}\{ \boldsymbol{A}^{T}\boldsymbol{L}(\boldsymbol{y} - \boldsymbol{b}) + \Lambda\boldsymbol{\mu} \} \\ cov[\boldsymbol{x}\vert \boldsymbol{y}] &= (\Lambda + \boldsymbol{A}^{T}\boldsymbol{LA} )^{-1} \end{align}$$  

이 결과를 다음과 같이 적용한다.  

$$\begin{align} \boldsymbol{x}&=\boldsymbol{w} \\ \boldsymbol{y}&=\boldsymbol{t} \\ \Lambda^{-1}&=\boldsymbol{S}_{0} \\ \boldsymbol{L}^{-1}&=\beta^{-1}\boldsymbol{I} \\ \boldsymbol{A}&=\Phi \\ \boldsymbol{\mu}&=\boldsymbol{m}_{0} \\ \end{align}$$  

따라서 

$$\begin{align} p(\boldsymbol{w}\vert \boldsymbol{t})&=\mathcal{N}(\boldsymbol{w}\vert \boldsymbol{m}_{N}, \boldsymbol{S}_{N}) \\ \boldsymbol{S}_{N}&=(\Lambda + \boldsymbol{A}^{T}\boldsymbol{LA})^{-1} \\ &=(\boldsymbol{S}_{0}^{-1} + \beta\Phi^{T}\Phi)^{-1} \\ \boldsymbol{S}_{N}^{-1}&=\boldsymbol{S}_{0}^{-1} + \beta\Phi^{T}\Phi \\ \boldsymbol{m}_{N}&=\boldsymbol{S}_{N}\{ \boldsymbol{A}^{T}\boldsymbol{L}(\boldsymbol{y} - \boldsymbol{b}) + \Lambda\boldsymbol{\mu} \} \\ &= \boldsymbol{S}_{N}\{ \Phi^{T}\beta\boldsymbol{I}\boldsymbol{t} + \boldsymbol{S}_{0}^{-1}\boldsymbol{m}_{0} \} \\ &= \boldsymbol{S}_{N}\{ \boldsymbol{S}_{0}^{-1}\boldsymbol{m}_{0} + \beta\Phi^{T}\boldsymbol{t} \} \end{align}$$ 

다음과 같은 사전확률을 사용하면 식이 단순화된다.  

$$p(\boldsymbol{w}\vert \alpha) = \mathcal{N}(\boldsymbol{w}\vert 0, \alpha^{-1}\boldsymbol{I})$$  

이 경우에 사후확률은  

$$\begin{align} p(\boldsymbol{w}\vert \boldsymbol{t}) &= \mathcal{N}(\boldsymbol{w}\vert \boldsymbol{m}_{N}, \boldsymbol{S}_{N}) \\ \boldsymbol{m}_{N}&=\beta\boldsymbol{S}_{N}\Phi^{T}\boldsymbol{t} \\ \boldsymbol{S}_{N}^{-1}&=\alpha\boldsymbol{I} + \beta\Phi^{T}\Phi  \end{align}$$

사후확률의 로그값은 다음과 같다.  

$$\ln p(\boldsymbol{w}\vert \boldsymbol{t}) = -\frac{\beta}{2}\sum_{n=1}{N}\{ \boldsymbol{t}_{n} - \boldsymbol{w}^{T}\phi(\boldsymbol{x}_{n}) \}^2 - \frac{\alpha}{2}\boldsymbol{w}^{T}\boldsymbol{w} + \text{const}$$  

- 예측분포  

새로운 입력 $$\boldsymbol{x}$$  가 주어졌을 때 $$\boldsymbol{t}$$  를 예측  

$$p(t\vert \boldsymbol{t}, \alpha, \beta) = \int p(t\vert \boldsymbol{w}, \beta)p(\boldsymbol{w}\vert \boldsymbol{t}, \alpha, \beta)d\boldsymbol{w}$$  

이전 결과들을 적용해보면  

$$\begin{align} p(t\vert \boldsymbol{x}, \boldsymbol{w}, \beta)&=\mathcal{N}(\boldsymbol{t}\vert \phi(\boldsymbol{x})^{T}\boldsymbol{w}, \beta^{-1}) \\ p(\boldsymbol{w}\vert \boldsymbol{t}, \boldsymbol{X}, \alpha, \beta)&=\mathcal{N}(\boldsymbol{w}\vert \boldsymbol{m}_{N}, \boldsymbol{S}_{N}) \\ \boldsymbol{x}&=\boldsymbol{w} \\ \boldsymbol{y}&=t \\ \boldsymbol{mu}&=\boldsymbol{m}_{N} \\ \Lambda^{-1}&=\boldsymbol{S}_{N} \\ \boldsymbol{A}&=\phi(\boldsymbol{x})^{T} \\ \boldsymbol{L}^{-1}&=\beta^{-1} \\  \end{align}$$  

$$\begin{align} p(t\vert \boldsymbol{x}, \boldsymbol{w}, \beta) &= \mathcal{N}(t\vert \boldsymbol{A\mu}+\boldsymbol{b},\boldsymbol{L}^{-1} + \boldsymbol{A}\Lambda^{-1}\boldsymbol{A}^{T}) \\ &= \mathcal{N}(t\vert \phi(\boldsymbol{x})^{T}\boldsymbol{m}_{N}, \beta^{-1}+\phi(\boldsymbol{x})^{T}\boldsymbol{S}_{N}\phi(\boldsymbol{x})) \end{align}$$  

# Appendix

## MathJax

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
