---
date: 2021-07-30 02:57
title: "NLP - Word Embedding"
categories: DevCourse2 NLP
tags: DevCourse2 NLP
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Word Embedding
- 단어의 의미를 어떻게 나타낼 것인가
  - 좋은 표현 방식: 단어간의 관계를 잘 표현할 수 있어야 함

## 단어의 의미
- 어근(lemma), 의미(sense)
  - mouse(N)
    - any of numerous small rodents...
    - a hand-operated device that controls a cursor...

## 동의어(synonyms)
- 문맥상 같은 의미를 가지는 단어들
  - big, large
- 동의어라고 해서 항상 그 단어로 대체할 수 있는 것은 아니다.
  - water, h2o

## 유사성
- 유사한 의미를 가진 단어들(동의어는 아닌)
  - car, bicycle
  - cow, horse
- 수작업으로 평가한 단어들의 유사도

    |word1| word2| similarity|
    |---|---|---|
    |vanish |disappear| 9.8|
    |belief |impression| 5.95|
    |muscle |bone| 3.65|
    |modest |flexible| 0.98|
    |hole |agreement| 0.3|

## 연관성 (Relatedness)
- Semantic field
- Semantic frame

### Semantic field
- 특정한 주제(topic)이나 영역(domain)을 공유하는 단어들
- hospitals
  - surgeon, nurse, hospital
- restaurant
  - waiter, menu, plate
- houses
  - door, roof, family

### Semantic frame
- 특정 행위에 참여하는 주체들의 역할에 관한 단어들
  - 예) 상거래라는 행위에 참여하는 주체들: buy, sell, pay 등의 단어는 같은 semantic frame을 공유하고 있고 그런면에서 관련된 단어라고 말할 수 있음

## 벡터로 의미 표현하기
- Ludwig Wittgenstein
  > "The meaning of a word is its use in the language"
- 단어들은 주변의 환경(주변 단어들의 분포)에 의해 의미가 결정됨
- 만약, 두 단어 A와 B가 거의 동일한 주변 단어들의 분포를 가지고 있다면 그 두 단어는 유사어이다.
- 예시
  - For example, suppose you didn’t know the meaning of the word ongchoi (a recent borrowing from Cantonese) but you see it in the following contexts:
    - (6.1) Ongchoi is **delicious sauteed with garlic**.
    - (6.2) Ongchoi is superb **over rice**.
    - (6.3) ...ongchoi **leaves** with salty sauces...
  - And suppose that you had seen many of these context words in other contexts:
    - (6.4) ...spinach **sauteed with garlic over rice**...
    - (6.5) ...chard stems and **leaves** are **delicious**...
    - (6.6) ...collard greens and other **salty** leafy greens
  - The fact that ongchoi occurs with words like rice and garlic and delicious and salty, as do words like spinach, chard, and collard greens might suggest that **ongchoi is a leafy green similar to these other leafy greens**.
- 따라서 단어의 의미를 분포적 유사성 (distributional similarity)을 사용해 표현하고자 한다.
  - 벡터를 사용해서 분포적 유사엉을 표현
- 벡터공간 내에서 비슷한 단어들은 가까이 있다.
    ![word-embedding-ex.png](\assets\images\word-embedding-ex.png){: .align-center .img-80}  
- 이렇게 벡터로 표현된 단어를 embedding 이라고 부름. 보통은 밀집벡터인 경우를 임베딩이라고 부름.
- 최근 NLP 방법들은 모두 임베딩을 사용해서 단어의 의미를 표현.

# 왜 임베딩을 사용하는가?
- 임베딩을 사용하지 않는 경우
  - 각 속성은 한 단어의 존재 유무
  - 학습데이터와 테스트데이터에 동일한 단어가 나타나지 않으면 예측 결과가 좋지 못함
- 임베딩을 사용하는 경우
  - 각 속성은 단어임베딩 벡터
  - 테스트데이터에 새로운 단어가 나타나도 학습데이터에 존재하는 유사한 단어를 통해 학습한 내용이 유효.

# 임베딩의 종류
- 희소벡터 (sparse vector): 대부분의 원소들이 0
  - tf-idf
  - vector propagation
- 밀집벡터 (dense vector)
  - word2vec
  - glove

# TF-IDF
## Term-document matrix
- 각 문서는 단어들의 벡터로 표현된다.(vector space model)
  ![Shakespeare-term-document-matrix.png](\assets\images\Shakespeare-term-document-matrix.png){: .align-center .img-80}  

## 문서 벡터의 시각화
![visualization-document-vectors.png](\assets\images\visualization-document-vectors.png){: .align-center .img-80}  

## 단어 벡터
![Shakespeare-term-document-matrix-words.png](\assets\images\Shakespeare-term-document-matrix-words.png){: .align-center .img-80}  

## Word-word 행렬(term-context 행렬)
- 주변 단어들의 빈도를 벡터로 표현
    ![word-word-matrix.png](\assets\images\word-word-matrix.png){: .align-center .img-80}  
    ![visualization-word-vectors.png](\assets\images\visualization-word-vectors.png){: .align-center .img-80}  

## 벡터의 유사도 계산하기
- 두 벡터의 각도를 가지고 계산
- cosine을 사용해서 벡터의 유사도를 계산할 수 있다.
- cosine 값이 작으면 두 벡터 사이의 각도가 큼
- For some applications we pre-normalize each vector, by dividing it by its length, creating a **unit vector** of length 1. Thus we could compute a unit vector from a by dividing it by $$\vert a\vert$$. For unit vectors, the dot product is the same as the cosine.

$$\text{cosine}(\vec{v},\vec{w}) = \frac{\vec{v}\cdot\vec{w}}{\vert\vec{v}\vert\vert\vec{w}\vert} = \frac{\sum\limits_{\rm i=1}^{\rm N}v_{i}w_{i}}{\sqrt{\sum\limits_{\rm i=1}^{\rm N}v_{i}^{2}}\sqrt{\sum\limits_{\rm i=1}^{\rm N}w_{i}^{2}}}$$

![cos-vector-similarity-ex.png](\assets\images\cos-vector-similarity-ex.png){: .align-center .img-80}  
![cos-vector-similarity-visual.png](\assets\images\cos-vector-similarity-visual.png){: .align-center .img-80}  

## TF-IDF: Weighing terms in the vector
- 단어의 빈도수를 그대로 사용할 때의 문제점
  - 자주 나타나는 단어들("the", "it", "they")은 의미를 구별하는 데 도움이 되지 않음
  - 어떻게 하면 이런 단어들의 부작용을 최소화할 수 있을까
- tf-idf
  - 다음과 같이 문서 d내의 단어 t의 새로운 가중치 값을 계산한다.
  - $$w_{t,d} = tf_{t,d}\times idf_{t}$$

### Term Frequency (tf)
- $$tf_{t,d} = \text{count}(t,d)$$
- $$tf_{t,d} = \log_{10}(\text{count}(t,d)+1)$$

### Document Frequency (df)
- $$df_{t}$$: 단어 t를 포함하는 문서들의 개수
- Inverse document frequency (idf)
  - $$idf_{t} = \log_{10}\left(\frac{N}{df_{t}}\right)$$
  - N은 전체 문서의 개수
  - ![tf-idf-table.png](\assets\images\tf-idf-table.png){: .align-center .img-40}  

### Vector representation by tf-idf
- 빈도수 기반 (term-frequency)
  - ![Shakespeare-term-document-matrix.png](\assets\images\Shakespeare-term-document-matrix.png){: .align-center .img-80}  
- tf-idf 사용
  - ![tf-idf-term-document-matrix.png](\assets\images\tf-idf-term-document-matrix.png){: .align-center .img-80}  
  - **good**의 경우 tf-idf 에서는 변별력이 없는 단어로 판단됨.
  - **wit** 의 경우 term-frequency 에서는 20으로, tf-idf에서는 $$\underbrace{\log_{10}(20+1)}_{tf}\times \underbrace{0.037}_{idf} = 0.049$$
  - 문서들간의 유사성을 훨씬 더 잘 표현하게 됨.

## Dense Vectors
- tf-idf vector
  - 길다( $$\vert V\vert$$ = 20,000 ~ 50,000)
  - 희소성(sparse, 대부분의 원소가 0)
- Word2vec, glove
  - 짧다(50 ~ 1,000)
  - 밀집성(dense, 대부분의 원소가 0이 아님)
- Dense vector가 선호되는 이유
  - 더 적은 개수의 학습파라미터를 수반
  - 더 나은 일반화 능력
  - 동의어, 유사어를 더 잘 표현

# Word2vec
- 주어진 단어 w를 인접한 단어들의 빈도수로 나타내는 대신, 주변 단어를 예측하는 분류기를 학습하면 어떨까?
  - 단어 w가 주어졌을 때 단어 c가 주변에 나타날 확률은?
- 우리의 관심은 이 예측모델의 최종예측값이 아니라 이 모델 내 단어 w의 가중치 벡터임
- self-supervision
  - 이 모델을 학습하기 위한 목표값은 이미 데이터내에 존재
  - 사람이 수동으로 레이블을 생성할 필요가 없다.

## Skip-gram
- skipgram 모델은 한 단어가 주어졌을 떄 그 주변 단어를 예측할 확률을 최대화하는 것이 목표.
- 즉, 단어들의 시퀀스($$w_{1},\ldots,w_{T}$$)가 주어졌을 떄, 다음 확률을 최대화하고자 한다.
  - $$\prod_{t=1}^{T}\prod_{-m\lt j\lt m, j\neq 0}p(w_{t+j}\vert w_{t})$$
  - ![Main-ideas-of-word2vec.png](\assets\images\Main-ideas-of-word2vec.png){: .align-center .img-80}
- 파라미터를 명시화해서 우도함수($$L$$)로 표현하면 다음과 같다.
  - $$L(\theta) = \prod_{t=1}^{T}\prod_{-m\lt j\lt m, j\neq 0}p(w_{t+j}\vert w_{t}l\theta)$$
- 파라미터 ($$\theta = \{W, C\}$$)는 두 개의 임베딩 행렬 W와 C를 포함한다. W를 목표(또는 입력) 임베딩 행렬, C를 상황(또는 출력) 임베딩 행렬이라고 부른다.
  - ![word2vec-parameters.png](\assets\images\word2vec-parameters.png){: .align-center .img-80}
- 하나의 목표 단어 $$w$$와 상황단어 $$c$$가 주어졌을 때 skipgram 모델은 다음과 같은 확률모델을 가정한다.
  - $$p(c\vert w;\theta) = \frac{\exp(u_{c}^{T}v_{w})}{\sum_{c^{\prime}\exp(u_{c^{\prime}}^{T}v_{w})}}$$
  - $$w$$에 대한 임베딩을 $$v_{w}$$, $$c$$에 대한 임베딩을 $$v_{c}$$, 이 두개의 vector의 dot product($$u_{c}^{T}v_{w}$$)를 구함(유사성)
  - normalize를 하기 위해서 vocabulary 에서 $$c$$외의 모든 단어($$c^{\prime}$$)에 대해서 dot product를 구해서 normalize
  - $$p(c\vert w;\theta) = \frac{\exp(u_{c}^{T}v_{w})}{\sum_{c^{\prime}\exp(u_{c^{\prime}}^{T}v_{w})}} \rightarrow \text{softmax regression}$$
  - $$c\in V$$이기 때문에 output이 dimension이 큰 경우임.(아주 큰 분류 문제를 푸는 것임)
- 여기서 $$x_w$$를 단어 $$w$$에 대한 one-hot vector라고 하면($$v_w$$를 하나의 입력속성 벡터로), $$v_w$$와 $$u_c$$는 다음과 같이 정의된다.
  - 굉장히 dimension이 큰 multiclass-classification 문제로 해석할 수 있음.
  - $$v_{w} = (x_{w}^{T}\boldsymbol{W})^{T},u_{c} = (x_{c}^{T}\boldsymbol{C})^{T}$$
  - $$(x_{w}^{T}\boldsymbol{W})^{T} = \boldsymbol{W}^{T}x_{w}$$
    - ![word2vec-onehot.png](\assets\images\word2vec-onehot.png){: .align-center .img-50}
    - $$v_w$$만 열벡터 형태로 나오게 되는 것
- 이 문제에 해당하는 신경망은 다음과 같다.
  - ![skipgram-nn-arch.png](\assets\images\skipgram-nn-arch.png){: .align-center}
  - hidden layer의 값이 $$v_w$$
  - $$\exp(u_{c}^{T}v_{w})$$에서 $$u_{c}^{T}v_{w} = x_{c}^{T}\boldsymbol{C}\boldsymbol{W}^{T}x_{w}$$ 인데
    - figure에서 output인 $$\boldsymbol{C}\boldsymbol{W}^{T}x_{w}$$는 dimension이 vocabulary 전체 사이즈($$\vert v\vert$$)와 같은 벡터가 하나 나옴.
    - 각각의 위치는 하나의 단어에 대응, 즉, 하나의 단어의 logit 값이 계산됨. ($$u_{c_{1}}^{t}v_{w}, u_{c_{2}}^{t}v_{w}, \ldots$$)
    - 즉, nn의 output 값은, 입력에서 주어진 단어 $$w$$와 하나의 출력단어 $$c$$의 관계를 dot product로 계산한 값임.
  - 최종적으로 취하는 것은 $$W$$ 행렬임.
    - $$W$$ 행렬은 각각의 단어에 대해서 하나의 embedding vector를 돌려주는 것.
    - $$W$$ 행렬을 최종적으로 얻는 것을 w2v 학습과정이라고 볼 수 있음.
  - 여기서 큰 문제는 `softmax crossentropy`에서 확률로 만들어주기 위해 normalize할때 $$c^{\prime}$$에 해당하는 합의 개수때문에 계산량이 많다는 점.
    - 즉, vocabulary 사이즈에 대한 issue가 있음. (학습이 느려질 수 있음)
    - 해결책: `Noise-constrative estimation`: Normalization constant($$p(c\vert w;\theta) = \frac{\exp(u_{c}^{T}v_{w})}{\sum_{c^{\prime}\exp(u_{c^{\prime}}^{T}v_{w})}}$$의 분모)를 하나의 파라미터로 학습한다. 이진분류문제에 해당하는 새로운 목표함수를 최적화 시킨다. 이렇게 해서 얻어지는 파라미터들이 원래 likelihood의 최적해를 근사한다는 것이 증명
    - 이것을 조금 더 단순화 시키면 `negative sampling`이 됨.
    - Word2vec은 negative sampling을 사용
    - Skipgram with negative sampling(SGNS)

## Noise-Constrative Estimation
- Noise-Constrative Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics(JMLR) [link](https://www.jmlr.org/papers/volume13/gutmann12a/gutmann12a.pdf)
- A fast and simple algorithm for training neural probabilistic language model (ICML) [link](https://arxiv.org/abs/1206.6426)
  - 저널논문을 볼것.
  - context h가 주어져 있을 때, 다음 단어 w를 예측하는 문제

### A fast and simple algorithm for training neural probabilistic language model
- $$P_{\theta}^{h}(w) = \frac{\exp(s_{\theta}(w,h))}{\sum_{w^{\prime}}\exp(s_{\theta}(w^{\prime},h))}$$
  - $$\theta$$는 모델의 parameter
    - w와 h의 임베딩 벡터들
  - $$s_{\theta}(w, h)$$ is the scoring function with parameters $$\theta$$ which quantifies the compatibility of word $$w$$ with context $$h$$.
    - 단어 w와 context h 사이의 관계를 모델링하는 parametric 함수.
    - 여기서 w는 예측을 위한 단어고 주어진 것은 context h임.
  - 역시 분모에 있는 normalize constant 계산량이 문제임.
- 핵심 아이디어
  1. 고차원의 multi-class classification 문제를 binary classification 문제로 전환 후 근사해를 구하는 것.
  2. $$P_{\theta}^{h}(w)$$의 분모를 parameter 화 후 데이터로부터 학습.
- 두 가지 분포를 가정
  - $$P_{d}^{h}(w)$$: h가 주어졌을 때 w의 확률 분포, 결국 이 확률을 가장 잘 근사시키는 모델을 만드는 것. (the distribution of words that occur after a particular context h)
  - $$P_{n}(w)$$: 노이즈 확률, context h와 상관없는 단어들의 확률분포 (a context-independent )
- Binary classification으로 전환하기 위해서
  - $$P_{d}^{h}(w)$$ &rarr; 1개의 샘플 추출 &rarr; positive set &rarr; $$c_1$$
  - $$P_{n}(w)$$ &rarr; k개의 샘플 추출 &rarr; negative set &rarr; $$c_2$$
    - $$t_w = 1 \text{if} w\in POS$$
    - $$t_w = 0 \text{if} w\in NEG$$
    - 하나의 단어가 POS에서 나왔을 지 NEG 에서 나왔을 지를 푸는 이진분류 문제로.
  - 주어진 상황에서 이진분류를 풀기 위한 Likelihood
    - $$\begin{aligned}
      L_{bin}^{h}(\theta) &= \sum\limits_{\rm w\in POS \cup NEG} \{t_{w}\ln p^{h}(C_1\vert w; \theta) + (1-t_{w})\ln p^{h}(C_2\vert w; \theta)\} \\\\
      &= \sum\limits_{\rm w\in POS}\ln p^{h}(C_1\vert w; \theta) + \sum\limits_{\rm w\in NEG}\ln p^{h}(C_2\vert w; \theta) \\\\
    \end{aligned}$$
      - $$p^{h}(C_1\vert w; \theta) = \frac{1}{1 + \exp(-\ln \frac{p_{\theta}^{h}(w)}{kp_{n}(w)})}$$
      - $$p^{h}(C_2\vert w; \theta) = \frac{1}{1 + \exp(\ln \frac{p_{\theta}^{h}(w)}{kp_{n}(w)})}$$
      - $$-L_{bin}^{h}(\theta)$$은 $$\ln \frac{p_{\theta}^{h}(w)}{kp_{n}(w)}$$을 logit으로 사용하는 logistic regression의 loss 이다.
      - 핵심 아이디어 중 2번째 분모를 parameter화하기 위해, $$P_{\theta}^{h}(w) = P_{\theta^{0}}^{h_{0}}(w)\exp(c^{h})$$라고 가정.
        - $$P_{\theta^{0}}^{h_{0}}(w)$$는 unnormalized 된 함수.
        - 여기에 새로운 parameter, 즉 분모에 해당하는 parameter를 곱하는 형태로 표현한다고 가정($$c^{h}$$: context h마다 하나의 parameter를 학습한다는 의미)
        - 이 $$c$$를 데이터를 통해 학습하겠다는 의미.
        - 로그, $$\ln P_{\theta}^{h}(w) = \ln P_{\theta^{0}}^{h_{0}}(w) + c^{h}$$
          - $$\ln P_{\theta^{0}}^{h_{0}}(w)$$ &rarr; $$S_{\theta^{0}}(w,h)$$
            - $$S_{\theta^{0}}(w,h)$$ 이 부분은 w2v 같은 경우 w,h를 나타내는 두개의 임베딩 벡터의 dot product를 사용가능.
          - $$\ln \frac{p_{\theta}^{h}(w)}{kp_{n}(w)}$$ &rarr; $$\ln p_{\theta}^{h}(w) - \ln kp_{n}(w)$$
            - $$\ln p_{\theta}^{h}(w) - \ln kp_{n}(w) = S_{\theta^{0}}(w,h) + c^{h} - \ln kp_{n}(w)$$
            - 결국 이 부분이 logistic regression 문제를 풀 때 loss 함수에 넣어야 함.
              - $$S_{\theta^{0}}(w,h) = v_{w}^{T}v_{h}$$
              - $$k$$는 하이퍼 파라미터
              - $$- \ln kp_{n}(w)$$ 노이즈 확률
- 이렇게 학습한 $$p_{\theta}^{h}(w)$$ &rarr; $$p_{d}^{h}(w)$$를 근사시킨다.
  - 원래 이 문제를 푸는 것이 softmax 계산 때문에 힘들었지만, 2가지 핵심 아이디어를 통해서 이진분류로 전환시켜서 normalize constant를 학습시킴을 통해서 더 쉬운 문제로 전환.

## Negative Sampling
- likelihood에서 logit으로 사용한 부분을 더 단순화 시킴
- $$L_{bin}^{h}(\theta)$$에 나타나는 $$\ln \frac{p_{\theta}^{h}(w)}{kp_{n}(w)}$$ 대신에, $$\ln p_{\theta}^{h}(w)$$만 사용
- 노이즈 확률을 무시함
- 이 경우 학습된 $$p_{\theta}^{h}(w)$$는 $$\frac{p_{d}^{h}(w)}{kp_{n}(w)}$$를 근사.
- 어떤 노이즈 확률분포를 사용하는가에 따라서 근사시키는 값이 달라짐(노이즈에 의존)
- NCE 의 경우 노이즈와 무관하게 원래의 확률분포를 학습
- w2v에서도 negative sampling을 사용한 이유가, 그 주변의 단어들을 정확하게 예측하는 것이 아니라, 어느 정도 reasonable하게 단어의 표현을 학습하는 것이기 때문에 나쁘지 않다.
- 다음과 같은 목표 함수를 사용한다.
  - $$\log \sigma(u_{c_{pos}}^{T}v_{w}) + \sum\limits_{\rm i=1}^{k}\log(1-\sigma(u_{c_{neg{i}}}^{T}v_{w}))$$
- 신경망의 구조는 그대로
- Normalization constant 를 계산할 필요가 없다.
- gradient 계산이 단순화 된다.

## 학습데이터 생성
![w2v-training-data.png](\assets\images\w2v-training-data.png){: .align-center}

## Word2vec 학습과정 요약
- $$\vert v\vert$$ 개의 $$d$$차원 임베딩을 랜덤하게 초기화
- 주변 단어들의 쌍을 positive example로 생성
- 빈도수에 의해 추출된 단어들의 쌍을 negative example 로 생성
- 위 데이터를 사용해 분류기 학습
- 학습된 임베딩 $$w$$가 최종 결과물

# Appendix
## MathJax
- bigger sum: $$\sum\limits_{\rm i=1}$$
```
$$\sum\limits_{\rm i=1}$$
```

## Reference
> Speech and Language Processing Chapter 6.8 Word2vec: <https://web.stanford.edu/~jurafsky/slp3/ed3book_dec302020.pdf>  
> Natural Language Processing with Deep Learning CS224N/Ling284 lecture 3 :<https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/lectures/lecture3.pdf>  
> Word2Vec (skip-gram model): PART 1 - Intuition: <https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b>  