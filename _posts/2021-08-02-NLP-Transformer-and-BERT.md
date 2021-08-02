---
date: 2021-08-02 14:28
title: "NLP - Transformer and BERT"
categories: DevCourse2 NLP
tags: DevCourse2 NLP
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# ML 모델 이해를 위한 팁
- 복잡한 ML 모델을 이해하기 위해서 다음과 같은 순서를 따라갈 것
  - 이 모델이 풀려고 하는 문제는 무엇인가
  - 추론 단계를 먼저 이해. "학습"된 모델이 어떤 "입력"을 받아어 어떤 "출력"을 계산하는지 파악할 것
  - 무엇이 학습되는가(모델 파라미터는 무엇인가), 모델 파라미터와 입력 그리고 그 둘의 함수를 구분할 것.
  - 어떻게 학습되는가(모델 파라미터를 어떻게 학습하는가), 어떤 에러함수를 사용하는 가

# NLP 와 딥모델
- NLP를 위한 딥모델들
  - RNN(LSTM 등)
  - TextCNN
  - Transformer
  - BERT (Transformer encoder)
  - GPT (Transformer decoder)

## RNN
![encoder_decoder2.png](\assets\images\encoder_decoder2.png){: .align-center}
- 문제점
  - 멀리 떨어진 단어들 간의 의존성을 모델링하기 어려움
  - 순차적 속성으로 인한 느린 속도

## TextCNN
![textcnn-arch.png](\assets\images\textcnn-arch.png){: .align-center}
- 속도는 빠름(conv 계산의 병렬화)
- 멀리 떨어진 단어들간의 의존성 어려움
- 나쁘지 않은 성능

## Transformer 모델
- Attention Is All You Need
  - No RNN, No convolution
  - 오직 attention으로만 단어의 의미를 문맥에 맞게 잘 표현할 수 있다
  - 병렬화 가능
  - BERT, GPT 등의 모델의 기반
### 이 모델이 풀려고 하는 문제는 무엇인가
  - seq2seq
  - 순차적입력에 대해 순차적 출력을 반환
  - 기계번역
  - 질의응답

### 추론 단계 이해
![the_transformer_3.png](\assets\images\the_transformer_3.png){: .align-center}
![The_transformer_encoders_decoders.png](\assets\images\The_transformer_encoders_decoders.png){: .align-center}
- The encoding component is a stack of encoders (the paper stacks six of them on top of each other – there’s nothing magical about the number six, one can definitely experiment with other arrangements). The decoding component is a stack of decoders of the same number.
![The_transformer_encoder_decoder_stack.png](\assets\images\The_transformer_encoder_decoder_stack.png){: .align-center}
- The encoders are all identical in structure (yet they do not share weights). Each one is broken down into two sub-layers:
![encoder_with_tensors.png](\assets\images\encoder_with_tensors.png){: .align-center}
- 하나의 단어들이 각각 단어 임베딩을 가지고 있음($$x_1$$),
- 이 임베딩이 self-attention 모듈의 입력으로 들어감
- self-attention 모듈을 통과하면서 다른 단어들과의 관계를 사용해서 그 단어의 의미를 명확하게 함.(self-attention은 해당 단어 주변의 단어들을 사용해 문맥을 활용함, 문맥화된 단어의 의미를 파악)
- 새로운 embedding이 self-attention 모듈을 통과하면서 출력으로 나오게 됨(이 결과는 주변의 모든 단어를 다 사용한 것)
- 이 출력 값(embedding)이 Feed Forward를 지나감, 각각의 단어들이 별개로 지나감(동일한 파라미터 사용, 다른 단어들과 의존성이 없기 때문에 병렬화 가능, 히자만 self-attention은 단어간의 관계를 봐야하기 때문에 병렬화는 힘듬)
- self-attention을 통과한 후 Feed Forward를 지나가게 하는 이유는 표현력을 높이기 위함이라고 함.(구체적으로 어떤것?)

#### 추론 단계 이해 - Self-Attention
![transformer_self-attention_visualization.png](\assets\images\transformer_self-attention_visualization.png){: .align-center}
- "The animal didn't cross the street because it was too tired"
- 여기서 "it"이 가리키는 단어는?
- 단어의 의미는 문맥에 의해 결정됨. 같은 단어라도 문맥에 의해 뜻이 달라짐
- 현재 단어의 의미(임베딩을 통해 표현되는)를 주변 단어들의 의미의 조합(weighted sum)으로 표현
- As we encode the word "it", one attention head is focusing most on "the animal", while another is focusing on "tired" -- in a sense, the model's representation of the word "it" bakes in some of the representation of both "animal" and "tired".
- ![transformer_self-attention-weighted-sum.png](\assets\images\transformer_self-attention-weighted-sum.png){: .align-center}
  - 이렇게 weighted sum을 활용해서 "it"의 embedding을 계산하고 나면, 관련있는 단어들의 임베딩 성분이 많이 포함되어 있게 됨.
  - 이런식으로 하면 문맥적의미를 포함하고 있는 것이기 때문에 좋은 성능을 기대할 수 있음.
  - RNN과 비교하면, RNN의 경우 문맥의 정보가 hidden state를 통해 전달되는데, 이것은 이전 단계의 정보만을 담고 있음.(unidirectional, 단방향)
  - Transformer는 bidirectional, 즉 문맥 전체를 고려할 수 있기 때문에 훨씬 의미가 풍부해짐

#### 추론 단계 이해 - Self-Attention - 단어 임베딩 벡터 중심으로
1. The first step in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word).
    ![transformer_self_attention_vectors.png](\assets\images\transformer_self_attention_vectors.png){: .align-center}
    - Multiplying $$x_1$$ by the $$WQ$$ weight matrix produces $$q_1$$, the "query" vector associated with that word. We end up creating a "query", a "key", and a "value" projection of each word in the input sentence.
    - $$q^T = x^{T}W^{Q}$$, $$k^{T} = x^{T}W^{k}$$, $$v^{T} = x^{T}W^{v}$$
    - 여기에서 입력은 $$x^T$$이고 모델 파라미터는 행렬들($$W^{q}, W^{k}, W^{v}$$) 임.
    - $$\underbrace{q^{T}}_{\text{함수값}} = \underbrace{x^{T}}_{\text{입력}}\underbrace{W^{q}}_{\text{모델 파라미터}}$$
    - 보통 x의 차원은 512, q의 차원은 64를 많이 사용함.
2. Self-attention score: The second step in calculating self-attention is to calculate a score.
    ![transformer_self_attention_score.png](\assets\images\transformer_self_attention_score.png){: .align-center}
    - "Thinking"의 score를 구할 때는, thinking의 쿼리벡터($$q_1$$)을 가지고 나머지 모든 단어들의 키벡터($$k_1, k_2$$)와의 dot product를 구함(자기자신 포함)
3. Divide the scores by 8 (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients.
4. Pass the result through a softmax operation. Softmax normalizes the scores so they’re all positive and add up to 1.
    - ![self-attention_softmax.png](\assets\images\self-attention_softmax.png){: .align-center}
    - 논문 상에는 embedding의 사이즈 64의 root값인 8로 나눠줌(gradient 계산을 안정적으로 하기 위해서)

    - softmax 로 변환(각각의 값이 확률로 됨)
    - ![attention-matrix-form.png](\assets\images\attention-matrix-form.png){: .align-center .img-40}
5. The fifth step is to multiply each value vector by the softmax score (in preparation to sum them up). 
6. The sixth step is to sum up the weighted value vectors.
    - ![self-attention-output.png](\assets\images\self-attention-output.png){: .align-center}
    - not embedding vector, its value vector.(weighted)
    - 자신의 가중치인 value vecto와 softmax를 곱한다.
      - $$z_1 = \text{softmax value}\times v_1 + \text{softmax value}\times v_2 = 0.88 \times v_1 + 0.12 \times v_2$$
    - ![attention-matrix-score.png](\assets\images\attention-matrix-score.png){: .align-center .img-40}


#### 추론 단계 이해 - Self-Attention - 행렬연산으로 표현
- x: 입력
- $$W^{q}, W^{k}, W^{v}$$: 모델 파라미터
- $$Q, K, V$$: 입력과 모델 파라미터에 관한 함수의 출력값
- The first step is to calculate the Query, Key, and Value matrices. We do that by packing our embeddings into a matrix X, and multiplying it by the weight matrices we’ve trained (WQ, WK, WV).
  - ![self-attention-matrix-calculation.png](\assets\images\self-attention-matrix-calculation.png){: .align-center}
- Finally, since we’re dealing with matrices, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer.
  - 행렬의 dimension 정리
    - s: sequence length, h:size_per_head(파라미터의 열의 개수, 64), d: input embedding size(x의 열의 개수, 512)
    - ![attention-matrix-all.png](\assets\images\attention-matrix-all.png){: .align-center}
  - ![self-attention-matrix-calculation-2.png](\assets\images\self-attention-matrix-calculation-2.png){: .align-center}
    - $$z = s \times h$$, h: size_per_head

#### 추론 단계 이해 - Multi-headed attention
- 다양한 attention matrix들을 반영하기 위한 방법. 블로그 오역 주의🚨
- cnn 에서 필터를 사용해서 여러 형태를 보기 위했던 것 처럼 학습능력 향상을 위해.
![transformer_attention_heads_qkv.png](\assets\images\transformer_attention_heads_qkv.png){: .align-center}
![transformer_attention_heads_z.png](\assets\images\transformer_attention_heads_z.png){: .align-center}
- This leaves us with a bit of a challenge. The feed-forward layer is not expecting eight matrices – it’s expecting a single matrix (a vector for each word). So we need a way to condense these eight down into a single matrix. How do we do that? We concat the matrices then multiple them by an additional weights matrix $$W^O$$.
  - ![transformer_attention_heads_weight_matrix_o.png](\assets\images\transformer_attention_heads_weight_matrix_o.png){: .align-center}
  - 행의 개수는 유지하되, 열의 개수를 다시 원래대로 만들어주는 $$W^{O}$$ 행렬을 곱해줌.
  - Multi-headed attention에서는 $$W^O$$ 가 추가적으로 학습해야되는 파라미터임.
- That’s pretty much all there is to multi-headed self-attention. It’s quite a handful of matrices, I realize. Let me try to put them all in one visual so we can look at them in one place
  - ![transformer_multi-headed_self-attention-recap.png](\assets\images\transformer_multi-headed_self-attention-recap.png){: .align-center}

### Self-Attention 구현
- <https://github.com/google-research/bert/blob/master/modeling.py>  
  - BERT 에서 Transformer의 attention을 사용하고 있음
<details><summary>Bert attention function</summary>
<script src="https://gist.github.com/marquis08/e551310b73a736f793d3fb834f047d3a.js"></script></details>





    


# Appendix
## MathJax
- bigger sum: $$\sum\limits_{\rm i=1}$$
```
$$\sum\limits_{\rm i=1}$$
```

## Reference
> Speech and Language Processing Chapter 9.4 self-attention network - Transformers: <https://web.stanford.edu/~jurafsky/slp3/ed3book_dec302020.pdf>  
> encoder decoder : <https://www.baeldung.com/cs/nlp-encoder-decoder-models>  
> A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification: <https://arxiv.org/abs/1510.03820>  
> The Illustrated Transformer: <https://jalammar.github.io/illustrated-transformer>  
> <https://ahnjg.tistory.com/57>