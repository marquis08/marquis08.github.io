---
date: 2021-07-02 13:32
title: "cs231n Assignment"
categories: DevCourse2 NN MathJax cs231n
tags: DevCourse2 NN MathJax cs231n
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# 1. 예제 따라하고, 개인 Github 정리하기
## CNN

- <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py> 사진 분류
- <https://pytorch.org/vision/stable/models.html 의 PyTorchVision을 참고하여 Pre-Trained AlexNet/VGG/ResNet/DenseNet> 성능 비교하기
- <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html> 의 전이학습

## RNN
- <https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#> 의 이름 분류
- <https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html> 의 문자 분류
- <https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html> 의 번역
- <https://pytorch.org/tutorials/beginner/transformer_tutorial.html> 의 TRANSFORMER 구조 번역

# 2. Multi-Layer Fully Connected Neural Networks

The notebook FullyConnectedNets.ipynb will have you implement fully connected networks of arbitrary depth. To optimize these models you will implement several popular update rules.

## Affine Layer
Affine transform (dot Product):  
Affine이라 함은 순전파에서 수행하는 행렬의 내적을 기하학에서 부르는 말로 Input 값과 Weight 값들을 행렬 곱하여 계산하고 거기에 편향(Bias)를 추가하여 출력값 Y를 최종적으로 반환

![backprop-matrices-1](/assets/images/backprop-matrices-1.png)  
![backprop-matrices-1](/assets/images/backprop-matrices-2.png)  

forward in code:  
```py
out = x.reshape(x.shape[0], -1)@w + b
```  

backward in code:
```py
dx = dout.dot(w.T).reshape(x.shape)
dw = x.reshape(x.shape[0], -1).T.dot(dout)
db = np.sum(dout, axis=0) # axis=0: column
```  

description
```py
"""
Inputs:
- dout: Upstream derivative, of shape (N, M)
- cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)
    - b: Biases, of shape (M,)

Returns a tuple of:
- dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
- dw: Gradient with respect to w, of shape (D, M)
- db: Gradient with respect to b, of shape (M,)
"""
```  
Check dims ❗❗  
- dx:
    - `dx = dout.dot(w.T).reshape(x.shape)`
    - `dout`: (10,5)
    - `w`: (6,5)
    - `dout.dot(w.T)`: (10,6)
    - `dout.dot(w.T).reshape(x.shape)`: (10,2,3)
- dw:
    - `dw = x.reshape(x.shape[0], -1).T.dot(dout)`
    - `x.shape[0]`: batch size (x.shape = (10,2,3))
    - `x.reshape(x.shape[0], -1)`: (10,6)
    - `dout`: (10,5)
- db:
    - shape should be equal to the dout's column(`dout: (10,5)`).
    - thus, `db = np.sum(dout, axis=0)`, sum by row-wise.
    - `np.sum(dout, axis=0)`: (5,)
- dout = `out` in image.  

## ReLU activation (Rectifier Linear Unit)
forward:  
For $$x > 0$$ the output is $$x$$, i.e. $$f(x) = \max(0,x)$$
backward:  
if $$x < 0$$, output is $$0$$. if $$x > 0$$, output is $$1$$.

forwrd in code:
```py
out = np.maximum(x, 0)
# or
# out = x * (x > 0)
```  

backward in code:  
```py
dout[x<0] = 0
dx = dout
```  

## Inline Question 1: 

We've only asked you to implement ReLU, but there are a number of different activation functions that one could use in neural networks, each with its pros and cons. In particular, an issue commonly seen with activation functions is getting zero (or close to zero) gradient flow during backpropagation. Which of the following activation functions have this problem? If you consider these functions in the one dimensional case, what types of input would lead to this behaviour?
1. Sigmoid
2. ReLU
3. Leaky ReLU

## Answer:
1. Sigmoid
    ![activation-gradient](/assets/images/activation-gradient.png)  

## Two-layer network

# 3. Batch Normalization

In notebook BatchNormalization.ipynb you will implement batch normalization, and use it to train deep fully connected networks.

# 4. Convolutional Neural Networks

In the notebook ConvolutionalNetworks.ipynb you will implement several new layers that are commonly used in convolutional networks.

# 5. Image Captioning with Vanilla RNNs

The notebook RNN_Captioning.ipynb will walk you through the implementation of vanilla recurrent neural networks and apply them to image captioning on COCO.

# Appendix
## References
> Affine transform: <https://ayoteralab.tistory.com/entry/ANN-11-%ED%99%9C%EC%84%B1%ED%99%94%ED%95%A8%EC%88%98s-Back-Propagation-Affine-Softmax>  
> Affine transform: <https://sacko.tistory.com/39>  
> explanation: <https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=tinz6461&logNo=221583992917>  