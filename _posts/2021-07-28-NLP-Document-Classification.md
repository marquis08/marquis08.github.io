---
date: 2021-07-28 15:13
title: "NLP - Document Classification"
categories: DevCourse2 NLP
tags: DevCourse2 NLP
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# 문서 분류 (Text Classification)
- 문서 분류란 입력으로 받아, 텍스트가 어떤 종류의 범주에 속하는지를 구분하는 작업
- 다양한 문서 분류 문제
  - 문서의 범주, 주제 분류
  - 이메일 스팸 분류
  - 감성 분류
  - 언어 분류

## 정의
input:  
- a document $$d$$
- a fixed set of classes $$ C =\{c_{1},\ldots, c_{j} \}$$

output: a predicted class $$c \in C$$  

## 문서 분류 방법들 - 규칙 기반 모델 (Rule based)
- 단어들의 조합을 사용한 규칙들을 사용
  - spam: black-list-address OR ("dollars" AND "you have been selected")
- Precision은 높지만 recall이 낮음
- Snorkel
  - 각각의 규칙을 "labeling function"으로 간주
  - Graphical model 의 일종인 factor graph 를 사용해서 확률적 목표값을 생성하는 generative model
  - 프로젝트 초기 labeled data가 부족하거나 클래스 정의 자체가 애매한 경우 (하지만 규칙을 생성하는 것은 쉽다고 가정)에 매우 유용한 방법
  - 확률적 목표값이 생성된 후엔 딥모델 등 다양한 모델을 사용가능

## 문서 분류 방법들 - 지도 학습
- input
  - a document $$d$$
  - a fixed set of classes $$ C =\{c_{1},\ldots, c_{j} \}$$
  - A training set of $$m$$ hand-labeled documents $$(d_1, c_1),\ldots, (d_m, c_m)$$

- output: a learned classifier $$y:d \rightarrow c$$

- 다양한 모델
  - Naive Bayes
  - Logistic regression
  - Neural networks
  - k-Nearest Neighbors
  - ...

# Naive Bayes Classifier (4.1)
- **Naive Bayes 가정**과 **Bag of Words 표현**에 기반한 간단한 모델

## Bag of Words 표현
- **Figure 4.1**: Intuition of the multinomial naive Bayes classifier applied to a movie review. **The position of the words is ignored** (the bag of words assumption) and we make use of the frequency of each word.
![naive-bayes-bow.png](\assets\images\naive-bayes-bow.png){: .align-center .img-60}  

## 수식화
- 문서 $$d$$와 클래스 $$c$$  

$$P(c\vert d) = \frac{P(d\vert c)P(c)}{P(d)}$$  

$$\begin{aligned}
  C_{MAP} &= \arg\max_{c\in C}P(c\vert d) & \text{MAP(maximum a posteriori) = most likely class} \\\\
  &= \arg\max_{c\in C}\frac{P(d\vert c)P(c)}{P(d)} & \text{Bayes' rule} \\\\
  &= \arg\max_{c\in C}P(d\vert c)P(c) & \text{dropping the denominator P(d)}
\end{aligned}$$  

> dropping the denominator $$P(d)$$: This is possible because we will be computing $$\frac{P(d\vert c)P(c)}{P(d)}$$ for each possible class. But $$P(d)$$ doesn’t change for each class  
> we are always asking about the most likely class for the same document $$d$$, which must have the same probability $$P(d)$$. Thus, we can choose the class that maximizes this simpler formula:

$$\begin{aligned}
  C_{MAP} &= \arg\max_{c\in C}\overbrace{P(d\vert c)}^{\text{likelihood}}\overbrace{P(C)}^{\text{prior}} \\\\
  &= \arg\max_{c\in C}\overbrace{P(x_1,\ldots, x_n\vert c)}^{\text{likelihood}}\overbrace{P(c)}^{\text{prior}} \\\\
  & \text{Without loss of generalization, we can represent a document d as a set of features } x_1,\ldots, x_n
\end{aligned}$$

> Unfortunately, still too hard to compute directly: without some simplifying assumptions, estimating the probability of every possible combination of features (for example, every possible set of words and positions) would require **huge numbers of parameters** and **impossibly large training sets**.  
> Naive Bayes classifiers therefore make **two simplifying assumptions**: **Bag of words**, **navie Bayes assumption**.

- Bag of words 가정: 위치는 확률에 영향을 주지 않는다.
- 조건부독립가정: 클래스가 주어지면 속성들은 독립적이다.

> The first is the **bag of words** assumption discussed intuitively above: we assume position doesn’t matter, and that the word “love” has the same effect on classification whether it occurs as the 1st, 20th, or last word in the document. Thus we assume that the features $$x_1,\ldots, x_n$$ only encode word identity and not position.  
> 
> The second is commonly called the **naive Bayes assumption**: this is the conditional independence assumption that the probabilities $$P( x_i\vert c)$$ are independent given the class c and hence can be ‘naively’ multiplied as follows:  
>  
> $$P(x_1,\ldots, x_n\vert c) = P(x_1\vert c)\cdot\ldots\cdot P(x_n\vert c)$$  

이렇게 단순화하게 되면 parameter의 개수가 줄어듬.  

from this:  

$$C_{MAP} = \arg\max_{c\in C}P(x_1,\ldots, x_n\vert c)P(c)$$  

to this:  

$$C_{NB} = \arg\max_{c_{j}\in C}P(C_j)\prod_{i\in \text{positions}}P(x_i\vert c_j)$$  


## NB 분류기는 입력값에 관한 선형모델임
- $$C_{NB} = \arg\max_{c_{j}\in C}P(C_j)\prod_{i\in \text{positions}}P(x_i\vert c_j)$$  


### 확률적 생성 모델 복습
- $$P(C_k\vert x) = \frac{P(x\vert C_k)P(C_k)}{\sum_{j}P(x\vert C_j)P(C_j)} = \frac{\exp(a_k)}{\sum_{j}\exp(a_j)}$$
- $$a_k = \ln P(x\vert C_k)P(C_k)$$
- 여기서 x가 이산적인 변수일 경우
- x는 하나의 문장을 의미(M개의 원소를 가짐, X_m이 L개의 상태를 가질 수 있음, L: vocab size)
- x의 원소 각각이 OHE라고 생각하면 됨.
- NB 가정에 의해서 likelihood를 다시 써보면(조건부 독립 가정때문에 아래의 식으로 표현됨),
- $$P(x\vert C_k) = \prod_{m=1}^{M}P(x_m\vert C_k)$$
- OHE 이기 때문에, 베르누이 확률분포에서 다뤘던 것을 이용
- $$P(x\vert C_k) = \prod_{m=1}^{M}P(x_m\vert C_k) = \prod_{m=1}^{M}\prod_{l=1}^{L}\mu_{kml}^{x_{ml}}$$
- 여기서 $$\mu_{kml} = P(x_{ml} = 1\vert C_k)$$ 임
  - m은 몇번째 단어인가를 나타냄. l은 L개의 상태 중 하나를 나타냄
  - 결국 $$\prod_{l=1}^{L}\mu_{kml}^{x_{ml}}$$은 l이 1이 되는 인덱스만 살아남고 나머지는 0이 됨.
- $$a_k = \ln P(x\vert C_k)P(C_k)$$에 $$P(x\vert C_k)$$를 대입하면
  - $$a_k = \ln P(C_k) + \sum_{m=1}^{M}\sum_{l=1}^{L} \underbrace{x_{ml}}_{\text{input}}\underbrace{\ln \mu_{kml}}_{\text{parameter}}$$ 이 됨.
- 이 확률적 생성모델이 가지는 decision surface는 $$a_k$$에 의해서 결정되는데, 이 $$a_k$$가 위의 식처럼 $$x$$에 관한 선형식으로 표현 되는 것.
- 하지만, 이 식은 NB와 완벽하게 일치하는 식은 아님.
- NB 같은경우 Bag of words 를 추가적으로 적용하면 간략화됨.
- $$P(x_{ml} = 1\vert C_k) = \mu_{kml}$$
  - $$x_{ml} = 1$$ &rarr; m 번째위치에 있는 단어가 V의 l번째 단어인 사건
  - NB에서, $$\mu_{kml} = \mu_{km^\prime l} (m\neq m^\prime)$$, 단어의 위치 m이 다르더라도 같다면 두개의 parameter값이 같다. (단어가 나타날 확률이 포지션과 관계없음)
    - 따라서 $$m$$는 필요가 없게 됨.
    - $$K(클래스 개수)\times \text{vocab size=L}$$개의 파라미터를 $$\mu_{kl}$$만 필요함. (기존에는 K*M*L 의 파라미터가 필요함)


### 확률적 생성 모델 예제
- "It was great" &rarr; x
- $$P(x\vert c_1)$$
  - $$x_m = (0,\ldots,1,\ldots,0))^T \rightarrow \text{OHE} :$$
- V: {"it", "was", "great", "bad"}
  - $$(1,0,0,0)^T \rightarrow \text{"bad"}$$
  - $$(0,1,0,0)^T \rightarrow \text{"great"}$$
  - $$(0,0,1,0)^T \rightarrow \text{"it"}$$
  - $$(0,0,0,1)^T \rightarrow \text{"was"}$$
- "It was great" 
  - $$x_1 = (0,0,1,0)^T$$ &rArr; $$x_11 = 0, x_12 = 0, x_13 = 1, x_14 = 0$$
  - $$x_2 = (0,0,0,1)^T$$
  - $$x_3 = (0,1,0,0)^T$$
- $$P(\text{"It was great"}\vert C_1) = \prod_{m=1}^{M}P(x_m\vert C_1) = P(x_1\vert C_1)P(x_2\vert C_1)P(x_3\vert C_1)$$
  - $$P(x_1\vert C_1) = \mu_{111}^{x_11} \times \mu_{112}^{x_12} \times \mu_{113}^{x_13} \times \mu_{114}^{x_14} \times$$
    - 여기서 $$x_11 = 0, x_12 = 0, x_13 = 1, x_14 = 0$$을 대입하면
    - 남는 것은 $$\mu_{113}$$
  - 나머지에 대해 모두 수행하면, $$P(\text{"It was great"}\vert C_1) = \mu_{113}\cdot\mu_{124}\cdot\mu_{132}$$
  - Bag of words을 적용하면, $$\mu_{13}\cdot\mu_{14}\cdot\mu_{12}$$

### 모델의 파라미터는 어떻게 학습해야 하나
- Log likelihood: $$\sum_{m=1}^{M}\sum_{l=1}^{L}x_{ml}\ln\mu_{kl}$$
- Constraint: $$\sum_{l=1}^{L}\mu_{kl} = 1$$
- $$\mu_{kl} = P(x_{l} = 1\vert C_k)$$
- Bigram Language Model 에서 MLE를 구하는 상황과 동일함.  
  - $$P(a\vert b) = \alpha_{ab}, \sum_{a\in V}\alpha_{ab} = 1$$

- MLE
  - prior
    - $$\hat{P}(c_j) = \frac{\text{doccount(C = c_j)}}{N_{\text{doc}}}$$
      - $$N_{\text{doc}}$$: the total number of documents
      - $$\text{doccount(C = c_j)}$$: the number of documents in training data with *class j*
  - $$\hat{P}(w_i\vert c_j)$$
    - $$\hat{P}(w_i\vert c_j) = \frac{\text{count}(w_i, c_j)}{\sum_{w\in V}\text{count}(w, c_j)}$$
    - $$\text{count}(w_i, c_j)$$: 문서들 중에 해당 클래스를 가진 문서들을 모아서 하나의 문서로 병합, 그 문서에 나타나는 해당 단어의 개수
    - $$\sum_{w\in V}\text{count}(w, c_j)$$: 하나의 문서로 만든 것에 있는 모든 단어의 개수.
- 문제점: zero 확률
  - Imagine we are trying to estimate the likelihood of the word “fantastic” given class positive, but suppose there are no training documents that both contain the word “fantastic” and are classified as positive. Perhaps the word “fantastic” happens to occur (sarcastically?) in the class negative. In such a case the proobability for this feature will be zero.
    - $$\hat{P(\text{"fantastic"}\vert \text{positive})} = \frac{\text{count("fantastic", positive)}}{\sum_{w\in V}\text{count(w, positive)}} = 0$$
  - Simplest Solution: Laplace(add-1) smoothing
    - $$\hat{P(w_i\vert c)} = \frac{\text{count}(w_i, c)+1}{\sum_{w\in V}(\text{count}(w, c)+1)} = \frac{\text{count}(w_i, c)+1}{\sum_{w\in V}(\text{count}(w, c))+\vert V\vert}$$
- 알고리즘
  - ![train-naive-bayes.png](\assets\images\train-naive-bayes.png){: .align-center .img-70} 


## Naive Bayes 분류기 예제
- ex
  ![naive-bayes-classifier-ex.png](\assets\images\naive-bayes-classifier-ex.png){: .align-center .img-70} 
- prior
  ![naive-bayes-classifier-ex-prior.png](\assets\images\naive-bayes-classifier-ex-prior.png){: .align-center .img-70} 
- drop `with` (out of vocabulary)
  - The word with doesn’t occur in the training set, so we drop it completely (as mentioned above, we don’t use unknown word models for naive Bayes).
- Likelihoods from training (applying laplace smoothing)
  - The likelihoods from the training set for the remaining three words “predictable”, “no”, and “fun”, are as follows, from Eq. 4.14 (computing the probabilities  or the remainder of the words in the training set is left as an exercise for the reader)
  ![naive-bayes-classifier-ex-likelihood.png](\assets\images\naive-bayes-classifier-ex-likelihood.png){: .align-center .img-70} 
- Scoring the test set
  - For the test sentence S = “predictable with no fun”, after removing the word ‘with’, the chosen class, via Eq. 4.9, is therefore computed as follows:
  ![naive-bayes-classifier-ex-score.png](\assets\images\naive-bayes-classifier-ex-score.png){: .align-center .img-70}
  - The model thus predicts the class **negative** for the test sentence.

## Naive Bayes 분류기 요약
- Not so naive
- 적은 학습데이터로도 좋은 성능
- 빠른 속도(training, inference)
- 조건부독립 가정이 실제 데이터에서 성립할 때 최적의 모델
- 문서 분류를 위한 베이스라인 모델로 적합




# Appendix
## MathJax
- overbrace: $$\overbrace{x + \cdots + x}^{\text{likelihood}}$$
```
$$\overbrace{x + \cdots + x}^{\text{likelihood}}$$
```

## Reference
> Speech and Language Processing Chapter 4 Naive Bayes and Sentiment Classification: <https://web.stanford.edu/~jurafsky/slp3/ed3book_dec302020.pdf>