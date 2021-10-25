---
date: 2021-10-21 14:10
title: "Optimizers"
categories: DL_basics
tags: DL_basics
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Optimizers
> [https://youtu.be/mdKjMPmcWjY](https://youtu.be/mdKjMPmcWjY)
> 

## Gradient Descent

- Update once every epoch
- $$\begin{align}\theta = \theta - \alpha \nabla_{\theta}J(\theta)\end{align}$$. 

## Stochastic Gradient Descent

- update every sample
    - Takes long
    - sensitive to each sample
- $$\begin{align}\theta = \theta - \alpha \nabla_{\theta}J(\theta)\end{align}$$. 

## Mini-batch Gradient Descent

- Decrease noise of SGD
- update every N-sample
- $$\begin{align}\theta = \theta - \alpha \nabla_{\theta}J(\theta)\end{align}$$. 

## SGD + Momentum

- Decrease noise of SGD
- $$\begin{align}v = \gamma v + \eta\nabla_{\theta}J(\theta)\end{align}$$. 
- $$\begin{align}\theta = \theta - \alpha v\end{align}$$. 
- parameters of a model tend to change in single direction as examples are similar.
    - pros
        - With this property, using momentum, a model learns faster by paying little attention to some samples.
    - Cons
        - Choosing samples blindly not guarantee the right direction.

## SGD + Momentum + Acceleration

- $$\begin{align}v = \gamma v + \eta\nabla_{\theta}J(\theta-\gamma v)\end{align}$$. 
- $$\begin{align}\theta = \theta - \alpha v\end{align}$$. 

## Adaptive Learning Rate Optimizers

- learn more along one direction than another

## Adagrad

- for every epoch $$t$$
    - for every parameter $$\theta_{i}$$
- $$\begin{align}\theta_{t+1, i} = \theta_{t, i} - \frac \eta {\sqrt{G_{t, ii}} + \epsilon}\nabla_{\theta_{t,i}}J(\theta_{t,i})\end{align}$$.
- $$\begin{align}G_{t, ii} = G_{t-1,ii} + \nabla^{2}_{\theta_{t,i}}J(\theta_{t,i})\end{align}$$. 
- or $$\begin{align}G_{t, ii} = \nabla^{2}_{\theta_{1,i}}J(\theta_{1,i}) + \nabla^{2}_{\theta_{2,i}}J(\theta_{2,i}) + \nabla^{2}_{\theta_{3,i}}J(\theta_{3,i}) + \cdots + \nabla^{2}_{\theta_{t,i}}J(\theta_{t,i})\end{align}$$ 
    - $$\begin{align}G_{t, ii}\end{align}$$ : Sum of squares of the gradients w.r.t $$\theta_{i}$$ until that point
- Problem:
    - G is monotonically increasing over iterations
        - LR will decay to a point where the parameter will no longer update (no learning)
        - $$\begin{align}\frac \eta {\sqrt{G_{t, ii}} + \epsilon}\end{align}$$ tends to 0
        - learns slower

## Adadelta

- for every epoch $$t$$
    - for every parameter $$\theta_{i}$$
- reduce the influence of the past squared gradients by introducing gamma weight to all of those gradients
    - reduces the effect exponentially
    - $$\begin{align}\theta_{t+1, i} = \theta_{t, i} - \frac \eta {\sqrt{E[G_{t, ii}]} + \epsilon}\nabla_{\theta_{t,i}}J(\theta_{t,i})\end{align}$$.
    - $$\begin{align}G_{t, ii} = \gamma G_{t-1,ii} + (1-\gamma)\nabla^{2}_{\theta_{t,i}}J(\theta_{t,i})\end{align}$$. 
    - $$\begin{align}\frac \eta {\sqrt{E[G_{t, ii}]} + \epsilon}\end{align}$$  not tend to 0

## Adam (Adadelta + momentum)

- for every epoch $$t$$
    - for every parameter $$\theta_{i}$$
- Adadelta + expected value of past gradients
    - $$\begin{align}\theta_{t+1, i} = \theta_{t, i} - \frac \eta {\sqrt{E[G_{t, ii}]} + \epsilon}E[g_{t,i}]\end{align}$$. 
    - $$\begin{align}G_{t, ii} = \gamma G_{t-1,ii} + (1-\gamma)\nabla^{2}_{\theta_{t,i}}J(\theta_{t,i})\end{align}$$. 
    - $$\begin{align}E[g_{t,i}] = \beta E[g_{t-1,i}] + (1-\beta)\nabla^{2}_{\theta_{t,i}}J(\theta_{t,i})\end{align}$$.
        - slow initially, pick up speed over time.
        - similar to momentum
    - different step size for different parameters
    - With momentum for every parameter, Adam leads to faster convergence
    - This is why Adam is de facto of optimizer of many projects.

![optimizer-equations.png](/assets/images/optimizer-equations.png)

> ref: <https://youtu.be/mdKjMPmcWjY>

