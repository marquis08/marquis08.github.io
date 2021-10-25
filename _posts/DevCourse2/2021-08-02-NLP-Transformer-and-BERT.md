---
date: 2021-08-02 14:28
title: "NLP - Transformer and BERT"
categories: DevCourse2 NLP DevCourse2_NLP
tags: DevCourse2 NLP DevCourse2_NLP
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
    - `attention_layer`
      - `from_tensor`: float Tensor of shape [batch_size, from_seq_length, from_width].
        - from_seq_length: 문장 혹은 문서의 길이(샘플 당 몇개의 단어가 들어있는지)
        - from_width: 하나의 단어당 몇개의 차원을 가지고 있는 지(단어를 나타내는 임베딩이 몇개의 차원을 가지고 있는지)
      - `to_tensor`: float Tensor of shape [batch_size, to_seq_length, to_width].
        - from_tensor 와 동일한 구성
      - `num_attention_heads`: int. Number of attention heads.
      - `size_per_head`: int. Size of each attention head. (변환된 임베딩의 차원)
      - `Returns`: float Tensor of shape [batch_size, from_seq_length, **num_attention_heads * size_per_head**].
        - `num_attention_heads * size_per_head`: z벡터들을 concat 한 사이즈($$h \times N$$).
      - from_width라는 작은 차원이 들어갔다가 확장되어서 return.
        - 앞에서는 W를 곱해서 다시 원래사이즈로 변환한다고 했음.(그 부분은 이 함수에는 포함되어 있지 않고, 이 함수를 호출하는 `transformer_model`이라는 다른 함수 안에 들어있음)
      - `query_layer`: Attention Matrix 계산시 첫번째였던 행렬 X에 W^{Q}를 곱하는 부분 ($$Q = XW^{Q}$$)
        ```py
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length
        #   T = `to_tensor` sequence length
        #   N = `num_attention_heads`
        #   H = `size_per_head`

        from_tensor_2d = reshape_to_matrix(from_tensor)
        to_tensor_2d = reshape_to_matrix(to_tensor)

        # `query_layer` = [B*F, N*H]
        query_layer = tf.layers.dense(
        from_tensor_2d,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))
        ```
        - F, T 는 encoder 부분에서 같아야함( F=T(encoder) )
        - decoder 부분에서는 다른 경우가 나옴
        - $$Q = \overbrace{X}^{\text{from_tensor}}\overbrace{W^{Q}}^{\text{query_layer}}$$
        - `from_tensor`: $$[B, F, d]$$ (d는 `from_width`)
          - 텐서 연산은 $$[B, F, d]$$ 인 텐서의 shape을 2차원으로 바꾼다 &rarr; $$[B\times F, d]$$
            - `from_tensor_2d = reshape_to_matrix(from_tensor)` 이 라인을 통해 수행함(custom 함수임)
              - <details><summary>reshape_to_matrix</summary><script src="https://gist.github.com/marquis08/8355516538ceae34bb91d68ca12e3cff.js"></script></details>
            - 이렇게 2차원 행렬로 변환후 $$W^{Q}$$를 곱해주는 연산을 해야하는데, 이것은 따로 정의하는 것이 아니라, tf에서 dense layer를 통해서 함.
              - 이 dense layer의 input: `from_tensor_2d`, output: `num_attention_heads * size_per_head`(dense layer에서 출력의 노드 개수)
              - ![dense-layer-wq.png](\assets\images\dense-layer-wq.png){: .align-center .img-40}
              - 이런 행렬에 해당하는 파라미터를 학습하게 되는 것.
          - $$Q$$ 가 `query_layer`에 해당. key와 value 역시 마찬가지임.
            - 이제 Attention Matrix 계산이 가능함.
      - `attention_scores`: attention 행렬 계산
        ```py
        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))
        ```
        - `query_layer`, `key_layer`으로 $$QK^{T}$$를 수행해야 함. `transpose_b=True`로 K만 Transpose하고 matmul로 계산하는 것임.
        - 이렇게 한 후 softmax를 계산하면 되는데
        - 그 전에 해야할 것이 있음
        - **Masking**
          - <details><summary>attention_mask</summary><script src="https://gist.github.com/marquis08/d58585ded6c3f2dde47b21ed66583ab9.js"></script></details>
          - 필요한 이유: 
            - 실제 단어의 개수는 샘플마다 다를 것임. 
            - 입력에서 사용한 `from_seq_length`는 고정된 것. 각각의 샘플이 가질 수 있는 최대의 값임.
            - 하지만 모든 샘플이 동일한 개수가 있는 것이 아님.
            - 부족한 부분은 padding으로 다른 특수기호로 채워넣는 것임.
            - attention을 계산하면서 이렇게 채워넣은 padding에 대해서 attention을 계산하면 의미가 없음. (실제 단어사이의 계산이 필요하니까)
          - 방법:
            - attention_mask: 어떠한 부분이 실제 input인지
            - 예:
              - $$[[1,1,1],[1,1,0]]$$: 첫 번째 샘플의 경우 세개의 단어 모두가 실제 있는 것이고, 두 번째 샘플의 경우 마지막 단어는 padding이라는 의미
              - 모든 단어의 조합을 계산해야 하기 때문에 3x3의 행렬을 만든다.
              - i 번째 단어와 j 번째 단어 사이의 attention을 계산할지 여부를 판단할 수 있다. (1이면 계산하고 0이면 계산하지 않는다.)
              - ![attention-mask-ex.png](\assets\images\attention-mask-ex.png){: .align-center .img-40}
                - 2번째 샘플의 경우 1번째 단어와 3번째 단어의 attention은 계산하지 않음.
              - 한번 더 과정을 거쳐서 attention 계산 부분을 0으로 masked 부분을 -10000.0 으로 만들어준다.
                - ![attention-mask-ex-2.png](\assets\images\attention-mask-ex-2.png){: .align-center .img-40}
              - 이렇게 계산된 부분을 $$QK^{T}$$에 더해주면 됨.
                - **softmax** 계산을 하면 음의 무한대에 해당하는 부분이 0에 가까운 부분으로 변환됨.
            - masking matrix를 만들어내는 부분
              - <details><summary>create_attention_mask_from_input_mask</summary><script src="https://gist.github.com/marquis08/fe9b94ef6af25d11d320c7a8d904ed0a.js"></script></details>
      - `attention_probs = tf.nn.softmax(attention_scores)`로 계산이 됨.
      - `context_layer = tf.matmul(attention_probs, value_layer)`는 $$ATT\times V$$의 연산임
      - masking 부분은 decoder 부분에서도 사용됨.
        - decoder 의 경우 현재 단어보다 앞의 단어만 집중해야 하기 때문에 이걸 활용해서 가능(뒤의 단어를 음의 무한대로)

#### 추론 단계 이해 - Positional encoding
- 단어의 순서를 어떻게 표현할 것인가
- 단어의 위치 정보를 넣어주는 것이 도움이 될 수 있음(특정 태스크에서-분류작업)
- 기본 아이디어는 포지션 자체도 숫자가 아니라 임베딩으로 표현하자.
- The intuition here is that adding these values to the embeddings provides meaningful distances between the embedding vectors once they’re projected into Q/K/V vectors and during dot-product attention.
- ![transformer_positional_encoding_vectors.png](\assets\images\transformer_positional_encoding_vectors.png){: .align-center}
- ![transformer_positional_encoding_example.png](\assets\images\transformer_positional_encoding_example.png){: .align-center}
- 학습할 때 보지 못했던 포지션이 나타날 수 있음
  - 임베딩을 따로 학습하는 것이 아니라 함수로 표현(함수로 계산)

#### 추론 단계 이해 - Residual
- ![transformer_resideual_layer_norm.png](\assets\images\transformer_resideual_layer_norm.png){: .align-center}
  - dashed line 으로 표시된 부분이 skip connection(residual connection)
    - Vanishing Gradient을 해결할 수 있는 방법 중 하나

#### 추론 단계 이해 - Encoder 종합
- ![transformer_resideual_layer_norm_3.png](\assets\images\transformer_resideual_layer_norm_3.png){: .align-center}
- encoder 간의 파라미터는 공유하지 않는다.


#### 추론 단계 이해 - Decoder
- ![transformer_decoding_1.gif](\assets\images\transformer_decoding_1.gif){: .align-center}
- encoder 에서 나온 tensor로 K행렬, V행렬을 구함(Q는 decoder에서 나타나는 각각의 번역들이 query로 쓰이기 때문)
- encoder 의 최종 출력은 K행렬, V행렬
  - 이 행렬들이 decoder에 전달되고 첫 번째 단어를 출력함
- ![transformer_decoding_2.gif](\assets\images\transformer_decoding_2.gif){: .align-center}
  - 첫 번째 단어가 출력이 되면(decoder 에서 나온)
  - 그 다음 input으로 사용됨
- ![transformer_decoder_output_softmax.png](\assets\images\transformer_decoder_output_softmax.png){: .align-center}
  - The Final Linear and Softmax Layer
    - The decoder stack outputs a vector of floats. How do we turn that into a word? That’s the job of the final Linear layer which is followed by a Softmax Layer.
    - The Linear layer is a simple fully connected neural network that projects the vector produced by the stack of decoders, into a much, much larger vector called a logits vector.
    - Let’s assume that our model knows 10,000 unique English words (our model’s “output vocabulary”) that it’s learned from its training dataset. This would make the logits vector 10,000 cells wide – each cell corresponding to the score of a unique word. That is how we interpret the output of the model followed by the Linear layer.
    - The softmax layer then turns those scores into probabilities (all positive, all add up to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.

### 모델학습
- 에러함수는? CrossEntropy
  - ![one-hot-vocabulary-example.png](\assets\images\one-hot-vocabulary-example.png){: .align-center .img-80}
  - ![transformer_logits_output_and_label.png](\assets\images\transformer_logits_output_and_label.png){: .align-center .img-80}
  - ![output_target_probability_distributions.png](\assets\images\output_target_probability_distributions.png){: .align-center .img-80}

# BERT
## 이 모델이 풀려고 하는 문제는 무엇인가
- Transfer Learning을 통해 적은 양의 데이터로도 양질의 모델(분류기 등)을 학습하는 것
- ![bert-transfer-learning.png](\assets\images\bert-transfer-learning.png){: .align-center}

## 추론단계 이해 - Fine-tuned 모델
- ![BERT-classification-spam.png](\assets\images\BERT-classification-spam.png){: .align-center}

## 추론단계 이해 - Pre-trained 모델 - 입력
- ![bert-input-output.png](\assets\images\bert-input-output.png){: .align-center}
- CLS 토큰을 문장의 제일 처음에 넣어줌.
- ![bert-encoders-input.png](\assets\images\bert-encoders-input.png){: .align-center}
- Transformer encoder와 동일

## 추론단계 이해 - Pre-trained 모델 - 출력
- ![bert-output-vector.png](\assets\images\bert-output-vector.png){: .align-center}
- 하나의 단어에 대해 embedding 출력

## 추론단계 이해 - Fine-tuned 모델 - 출력
- ![bert-classifier.png](\assets\images\bert-classifier.png){: .align-center}
- self-attention이 다 발생하고 있기 때문에, 첫 번째 토큰에 대해서도 자기자신에 대한 내용만 있는 것이 아니라, input에 사용된 모든 단어들간의 관계로 표현되어있기 때문에, 모든 단어들의 정보를 가지고 있기 때문에 나머지 단어들에 대한 임베딩이 필요없음.
- classifier에는 특수 토큰(CLS)의 임베딩을 인풋으로 사용.
- 특수 토큰(CLS)의 임베딩 위에 원하는 classifier 모델을 쌓으면 됨. (추가 데이터로 학습해야 하는 부분)

## 모델 학습 - Pre-trained 모델
- BERT 모델이 Transformer 를 사용하고 있지만, decoder 부분이 없기 때문에 encoder만으로 어떻게 학습해야 할까
- Masked Language model
  - 주어진 input 단어들 중에 몇 개의 단어를 숨김
  - BERT 의 출력을 통해 masked 된 단어를 예측하도록 함
  - 학습을 하는 것은 transformer 안에 있는 모델들의 parameter 뿐만 아니라 input으로 주어지는 토큰들의 임베딩도 학습이 가능함.
  - BERT는 input이 단어가 아니라 sub word 단위로 학습함 (wordpiece tokenizer 사용)
  - ![BERT-language-modeling-masked-lm.png](\assets\images\BERT-language-modeling-masked-lm.png){: .align-center}

## 모델 학습 - Fine-tuned 모델
- Pre-trained 모델의 파라미터를 기초로 새로운 작업(분류 같은)을 위한 **적은 양의 데이터를 사용**해 파라미터를 업데이트 함.
- ![bert-tasks.png](\assets\images\bert-tasks.png){: .align-center}

## BERT - 응용
- BERT 의 output을 다른 모델의 input으로 사용 가능하다.
- 앞에서는 CLS 토큰을 통해서만 분류 task에 사용
- contextualized word embeddings: pretrained된 BERT 모델의 출력을 그대로 하나의 표현(문서전체의 표현)으로 사용
- ![bert-contexualized-embeddings.png](\assets\images\bert-contexualized-embeddings.png){: .align-center}
- 어떤 임베딩을 사용해야 제일 좋을까 라는 실험을 했음
  - ![bert-feature-extraction-contextualized-embeddings.png](\assets\images\bert-feature-extraction-contextualized-embeddings.png){: .align-center}
- Data Augmentation
  - 클래스를 바꾸지 않는 범위 안에서 입력을 변환. 학습데이터를 확장시킴. 더 나은 일반화 성능 기대
  - 이미지의 경우: shift, filp, resize, rotate
  - 텍스트는?
- BERT
  - 문서 D와 클래스 c가 주어졌을 때, D의 단어들을 랜덤하게 mask한 다음 BERT 를 사용해서 예측하고 그 결과를 D'로 학습데이터에 추가한다(클래스 c와 함께)
  - GPT 모델을 사용해서 비슷한 방식으로 학습데이터를 확장할 수 있다.

# Appendix
# BERT Transformer
<details><summary>Bert attention function</summary>
<script src="https://gist.github.com/marquis08/e551310b73a736f793d3fb834f047d3a.js"></script></details>


## MathJax
- bigger sum: $$\sum\limits_{\rm i=1}$$
```
$$\sum\limits_{\rm i=1}$$
```

## Gist embedding
<script src="https://gist.github.com/marquis08/b025c83062e84d72ccd3c4276fa9bc5f.js"></script>  


## Reference
> Speech and Language Processing Chapter 9.4 self-attention network - Transformers: <https://web.stanford.edu/~jurafsky/slp3/ed3book_dec302020.pdf>  
> encoder decoder : <https://www.baeldung.com/cs/nlp-encoder-decoder-models>  
> A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification: <https://arxiv.org/abs/1510.03820>  
> The Illustrated Transformer: <https://jalammar.github.io/illustrated-transformer>  
> illustrated-bert:<https://jalammar.github.io/illustrated-bert/>
> transformer 한글정리: <https://ahnjg.tistory.com/57>