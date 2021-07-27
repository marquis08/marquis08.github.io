---
date: 2021-07-27 01:35
title: "NLP - Language Model"
categories: DevCourse2 NLP
tags: DevCourse2 NLP
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# 언어모델
- 다음 문장 다음에 이어질 단어는?
  - Please turn your homework
    - `in` or `the` ?
- 다음 두 문장 중 나타날 확률이 더 높은 것은?
  - all of a sudden I notice three guys standing on the sidewalk.
  - on guys all I of notice sidewalk three a sudden standing the.
- 목표: 문장이 일어날 확률을 구하는 것
- 왜 필요한가?
  - 기계번역 (machine translation)
    - P(**high** winds tonight) > P(**large** winds tonight)
  - 맞춤법 검사 (spell correction)
    - The office is about fifteen **minuets** from my house
      - P(about fifteen **minutes** from) > P(about fifteen **minuets** from)
  - 음성인식 (speech recognition)
    - P(I saw a van) >> P(eyes awe of an)
- 언어모델: 연속적인 단어들 (sequence of words)에 확률을 부여하는 모델
  $$P(W) = p(w_{1}, w_{2}, ldots, w_{n})$$
- 관련된 일: 연속적인 단어들이 주어졌을 떄 그 다음 단어의 확률을 구하는 것
  $$P(w_{n}\vert w_{1}, w_{2}, ldots, w_{n-1})$$

## P(W) 구하기
- 결합확률 구하기
  $$P(\text{its, water, is, so, transparent, that})$$
- chain rule 을 사용해보자

### Chain rule
- 조건부 확률
  - $$P(B\vert A) = \frac{P(A, B)}{P(A)}$$  
  - $$P(A, B) = P(A)P(B\vert A)$$
- 두 개 이상의 확률변수들의 경우
  - $$P(A, B, C, D) = P(A)P(B\vert A)P(C\vert A,B)P(D\vert A, B, C)$$  
  - 풀이
    - 
    $$\begin{aligned} P(A, B, C, D) &= P(A, B, C)\ P(D\vert A, B, C) \\
    &= P(A, B)\ P(C\vert A, B)\ P(D\vert A, B, C) \\
    &= \ldots \\
    \end{aligned}$$
- 일반적인 경우
  - $$P(x_{1},x_{2},\ldots,x_{n}) = P(x_{1})\ P(x_{2}\vert x_{1})\ P(x_{3}\vert x_{1}, x_{2})\ldots \ P(x_{n}\vert x_{1},\ldots,x_{n-1}) $$
- $$P(w_{1},\ldots, w_{b}) = \prod_{i}P(w_{i}\vert w_{1}w_{2}\ldots, w_{i-1})$$
  - $$P(\text{"its water is so transparent"}) = P(\text{its})\times P(\text{water}\vert \text{its})\times P(\text{is}\vert \text{its water})\times P(\text{so}\vert \text{its water is})\times P(\text{transparent}\vert \text{its water is so})$$
- 조건부 확률 $$P(w\vert h)$$
  - $$P(\text{the}\vert \text{its water is so transparent that}) = \frac{\text{Count}(\text{its water is so transparent that the})}{\text{Count}(\text{its water is so transparent that})}$$
  - 문제는?
    - 가능한 문장의 개수가 너무 많음
    - 이것을 계산할 수 있는 충분한 양의 데이터를 가지지 못할 것임.
- Markov Assumption
  - "한 단어의 확률은 그 단어 앞에 나타나는 몇 개의 단어들에만 의존한다"라는 가정
    - $$P(\text{the}\vert \text{its water is so transparent that})\approx P(\text{the}\vert \text{that})$$
    - $$P(\text{the}\vert \text{its water is so transparent that})\approx P(\text{the}\vert \text{transparent that})$$
  - $$P(w_{i}\vert w_{1}\ldots, w_{i-1}) \approx \prod_{i}P(w_{i}\vert w_{i-k}\ldots, w_{i-1})$$
    - 바로 직전 $$k$$개의 단어에만 의존하는 조건부확률로 근사하겠다.
  - $$P(w_{1},\ldots, w_{n}) \approx \prod_{i}P(w_{i}\vert w_{i-k}\ldots, w_{i-1})$$
    - 단순화 시키면 곱의 형태로 나타나게 됨.
- Unigram 모델
  - $$P(w_{1},\ldots, w_{n}) \approx \prod_{i}P(w_{i})$$
  - 이 모델로 생성된 문장 예제들
    - fifth, an, of, futures, the, an incorporated, a, a, the, inflation, most, dollars, quarter, in, is, mass
- Bigram 모델
  - $$P(w_{i}\vert w_{1}\ldots, w_{i-1}) \approx P(w_{i}\vert w_{i-1})$$
  - 바로 직전의 단어에 대해서 조건부확률
- N-gram 모델
  - 이것을 trigrams, 4-grams, 5-grams로 확장할 수 있음
  - 멀리 떨어진 단어들간의 관계를 완벽하게 모델링하진 못한다.
  - 하지만 많은 경우 n-gram 만으로도 좋은 결과를 얻을 수 있음

# Bigram 확률 계산
## Maximum likelihood estimation
$$P(w_{i}\vert w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_{i})}{\text{Count}(w_{i-1})}$$  
$$P(w_{i}\vert w_{i-1}) = \frac{c(w_{i-1}, w_{i})}{c(w_{i-1})}$$  

## 왜 이것이 MLE 인가
문장이 여러개가 주어진 Data: D  
parameter: $$\theta$$  
parameter 가 주어졌을 때 data가 나타날 확률:   

$$P(D\vert \theta)$$  

## MLE 유도
$$P(D\vert \theta)$$  

우선, $$\theta$$에 대해 정의를 해야함.  

두 개의 단어, a와 b가 주어졌을 때.  

$$P(a\vert b) = \alpha_{ab}$$: 단어 b가 주어졌을 때 그 다음에 나타나는 단어 a의 확률  

$$P(a) = \beta_{a}$$: 단어 a의 확률  

$$D = \left\{W_{1},\ldots, W_{N} \right\}$$: D = 문장 들의 집합. $$W_{i}$$ = 문장  

> 문장일 경우 대문자, 단어일 경우 소문자로 사용할 것임.  

$$P(D\vert \theta) = \prod_{i=1}^{N}P(W_{i})$$  

여기서 $$\theta$$는 $$\left\{\alpha_{ab}, \beta_{a}\right\}$$를 다 포함한 것  

두 문장을 예로 들면,  

$$W_{1} = \text{"b a b"}$$  
$$W_{2} = \text{"a b a c"}$$  

$$\begin{aligned}
  P(D\vert \theta) &= P(W_{1})P(W_{2}) \\\\
  &= P(\text{"b a b"})P(\text{"a b a c"}) \\\\
  &= \beta_{b}\alpha_{ab}\alpha_{ba}\ \beta_{a}\alpha_{ba}\alpha_{ab}\alpha_{ca} \\\\
\end{aligned}$$  

> $$P(\text{"b a b"}) = \beta_{b}\alpha_{ab}\alpha_{ba}$$  
> $$P(\text{"a b a c"}) = \beta_{a}\alpha_{ba}\alpha_{ab}\alpha_{ca}$$  

로그를 적용해보면,  
$$\begin{aligned}
  \ln P(D\vert \theta) &= 2\ln \alpha_{ab} \\ &+ 2\ln \alpha_{ba} \\  &+ \ln\alpha_{ca} \\  &+ \beta_{a} \\  &+ \beta_{b}
\end{aligned}$$  

확률로 만드는 제약조건,  

$$\sum_{a\in V}\alpha_{ab} = 1$$  

b 는 고정이고 a 값만을 Vocabulary에 속해있는 모든 단어들에 대해서 더하는 것.  

$$\begin{aligned}
  \sum_{a\in V}P(\text{a|b}) &= \sum_{a\in V}\frac{P(a, b)}{P(b)} \\\\
  &= \frac{1}{P(b)}\sum_{a\in V}P(a, b)\ \text{# 결합확률(P(a, b))을 a에 관해서 합하게 되면 b에 관한 Marginal Probability가 되는 것} \\\\
  &= \frac{1}{P(b)}P(b) \\\\
  &= 1
\end{aligned}$$

따라서 이러한 제약조건이 Parameter($$\theta$$)에 적용되어야 함.  

이 조건 하에서 $$P(D\vert \theta) = \prod_{i=1}^{N}P(W_{i})$$을 최대화 시키는 방법을 찾아야 함.  

라그랑주 방법을 사용하면 쉽게 해결됨.  

$$\alpha_{ab} = C(b, a)\ln\alpha_{ab} + \lambda(\sum_{a^{\prime}\in V}\alpha_{a^{\prime}b} - 1)$$  
- 이 식이 최대화되는 $$\alpha_{ab}$$ 를 구하면 됨.
- $$C(b, a)$$: "b a" 가 나오는 빈도  

$$\frac{\partial}{\partial\alpha_{ab}}\ln_{ab} = \frac{C(b,a)}{\alpha_{ab}} + \lambda = 0$$  

0으로 놓고 풀면  

$$\lambda \alpha_{ab} = -C(b,a)$$  

이렇게 되고,  

모든 a 에 관해서 양변에 추가를 하면  

$$\sum_{a\in V}\lambda\alpha_{ab} = \sum_{a\in V}-C(b,a)$$  

$$\lambda\sum_{a\in V}\alpha_{ab} = -\sum_{a\in V}C(b,a)$$  

여기서 $$\sum_{a\in V}\alpha_{ab} = 1$$ 이므로, 좌변은 lambda만 남는다.  

$$\lambda = -\sum_{a\in V}C(b,a)$$  

이 값을 다시 위의 식($$\frac{\partial}{\partial\alpha_{ab}}\ln_{ab} = \frac{C(b,a)}{\alpha_{ab}} + \lambda = 0$$)에 대입  

$$\alpha_{ab} = \frac{C(b,a)}{\sum_{a\in V}C(b,a)}$$  

이 식은 앞에서 MLE 처음에 언급한 $$P(w_{i}\vert w_{i-1}) = \frac{c(w_{i-1}, w_{i})}{c(w_{i-1})}$$과 동일한 것을 알 수 있음.  

$$P(a) = \beta_{a}$$을 문장시작 기호에 대해서도 정의하면,  
$$\text{<s>: 문장 시작 기호 라고 했을때,  }\ \alpha_{a<s>}$$ 로 사용할 수 있다.  


## 예제
$$P(w_{i}\vert w_{i-1}) = \frac{c(w_{i-1}, w_{i})}{c(w_{i-1})}$$  
```
<s> I am Sam</s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>
```  
$$\begin{aligned}
P(I\vert <s>) = \frac{2}{3} = 0.67 & P(Sam\vert <s>) = \frac{1}{3} = 0.33 & P(am\vert I) = \frac{2}{3} = 0.67 \\
P(<s>\vert Sam) = \frac{1}{2} = 0.5 & P(Sam\vert am) = \frac{1}{2} = 0.5 & P(do\vert I) = \frac{1}{3} = 0.33 \\
\end{aligned}$$

- Bigram 빈도수
![bigram-freq.png](\assets\images\bigram-freq.png){: .align-center .img-80}  
- Bigram 확률
![bigram-probs.png](\assets\images\bigram-probs.png){: .align-center .img-80}  

$$\begin{aligned}
P(\text{<s> i want english food </s>}) &= P(\text{i|<s>})P(\text{want|i})P(\text{english|want})P(\text{food|english})P(\text{</s>|food})\\
&= 0.25\times 0.33\times 0.0011\times 00.5\times 00.68\\
&= 0.000031\\
\end{aligned}$$

## 모델평가
- 외재적 평가 (extrinsic evaluation)
  - 언어모델은 일반적으로 그 자체가 목표이기보다 특정 과제르 ㄹ위한 부분으로서 쓰여지게 됨
  - 따라서 언어모델이 좋은지 판단하기 위해선 그 과제의 평가지표를 사용하는 경우가 많음
  - 예를 들어, 맞춤법 검사를 위해서 두 개의 언어모델 A, B를 사용한다고 할 때
    - 각 모델을 사용해서 얼마나 정확하게 맞춤법 오류를 수정할 수 있는 지 계산
    - 정확도가 높은 언어모델을 최종적으로 사용
- 내재적 평가 (intrinsic evaluation)
  - 외재적 평가는 시간이 많이 걸리는 단점
  - 언어모델이 학습하는 확률자체를 평가 못함: perplexity(*3.2.1 Perplexity in SLP book*)
  - 이 기준으로 최적의 언어모델이 최종 과제를 위해서는 최적이 아닐 수도 있음
  - 하지만 언어모델의 학습과정에 버그가 있었는지 빨리 확인하는 용도로 사용 가능.

## Perplexity
- 좋은 언어모델이란?
  - 테스트 데이터를 높은 확률로 예측하는 모델
  - perplexity: 확률의 역수를 단어의 개수로 정규화한 값
  - perplexity를 최소화하는 것이 확률을 최대화 하는 것
- 
  $$\begin{aligned}
  PP(W) &= P(w_{1}w_{2}\ldots w_N)^{\frac{1}{N}} \\
  &= \sqrt[N]{\frac{1}{P(w_{1}\ldots w_{N})}}
  \end{aligned}$$
- use chain rule
  - $$PP(W) = \sqrt[N]{\prod_{i=1}^{N}\frac{1}{P(w_{i}\vert w_1\ldots w_{i-1})}}$$
- bigram perplexity
  - $$PP(W) = \sqrt[N]{\prod_{i=1}^{N}\frac{1}{P(w_{i}\vert w_{i-1})}}$$


# Appendix
## Reference
> Speech and Language Processing : <https://web.stanford.edu/~jurafsky/slp3/ed3book_dec302020.pdf>