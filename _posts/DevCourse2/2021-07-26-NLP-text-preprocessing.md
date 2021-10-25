---
date: 2021-07-26 15:10
title: "NLP - text preprocessing"
categories: DevCourse2 NLP DevCourse2_NLP
tags: DevCourse2 NLP DevCourse2_NLP
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Intro
- 자연어의 의미를 컴퓨터로 분석해서 특정작업을 위해 사용할 수 있도록 하는 것
- 응용분야
  - 기계번역
  - 감성분석
  - 문서분류
  - 질의응답시스템
  - 챗봇
  - 언어생성
  - 음성인식
  - 추천시스템

# Resource
- Speech and Language Processing <https://web.stanford.edu/~jurafsky/slp3/>
- Hands-On Machine Learning with Scikit-learn, Keras, and TensorFlow (2nd Edition)

# 단어 (Word)
- 다음 문장은 몇 개의 단어를 가지고 있나?
  - He stepped out into the hall, was delighted to encounter a water brother.
  - 문장부호를 단어에 포함시켜야 할까
- 구어체 문장의 경우
  - I do uh main- mainly business data processing
  - Fragments(깨진 단어), filled pauses(uh, um)
  - 음성 분석을 할 경우에는 빼지 않는 것이 도움이 될수 있음
- "Seuss's cat in the hat is different from other cats!"
  - 표제어(lemma): 여러 단어들이 공유하는 뿌리단어
  - 단어형태(wordform): 같은 표제어를 공유하지만 다양한 형태를 가질 수 있음
  - cat과 cats: 두 가지 형태를 가지고 있지만 동일한 표제어 cat을 공유

- Vocabulary: 단어의 집합
- Type: Vocabulary의 한 원소
- Token: 문장 내에 나타나는 한 단어(an instance of a type in running text)
- They picnicked by the pool, then lay back on the grass and looked at the stars.
  - 16 tokens
  - 14 types(the 가 반복됨)

# 말뭉치 (Corpus)
- 하나의 말뭉치는 일반적으로 대용량의 문서들의 집합이다.
- 말뭉치의 특성은 아래의 요소들에 따라 달라지게 됨
  - 언어
  - 방언
  - 장르(뉴스, 소설, 과학기술문서, 위키피디아, 종교문서 등)
  - 글쓴이의 인구통계적 속성(나이, 성별, 인종 등)
- 다양한 말뭉치에 적용할 수 있는 NLP 알고리즘이 바람직함.

# 텍스트 정규화
- 모든 자연어 처리는 텍스트 정규화를 필요로 한다.
  - 토큰화 (tokenizing words)
  - 단어 정규화 (normalizing word formats)
  - 문장 분절화 (segmenting sentences)

# Unix 명령으로 간단하게 토근화하기
- 텍스트 파일 안에 있는 단어들 토근화
    ```sh
    tr -sc 'A-Za-z' '\n' < hamlet.txt
    ```
    - Hamlet
    - by
    - William
    - Shakespeare
    - ...
- 빈도수로 정렬
    ```sh
    tr -sc 'A-Za-z' '\n' < hamlet.txt | sort | uniq -c | sort -n -r
    ```
    - 931 the
    - 697 and
    - ...
- 소문자로 변환해서 정렬
    ```sh
    tr 'A-Z''a-z'< hamlet.txt | tr -sc 'a-z' '\n' | sort | uniq -c | sort -n -r
    ```
    - 1090 the
    - 974 and
    - ...

# 문제점들
- 문장부호들을 항상 무시할 수는 없다.
  - 단어 안에 나타나는 문장부호들: m.p.h, AT&T, Ph.D.
  - 화폐단위, 날짜, URLs, hashtags, 이메일 주소
  - 문장부호가 단어의 의미를 명확하게 하는 경우는 제외시키지 않는 것이 좋다.
- 접어(clitics): 다른 단어에 붙어서 존재하는 형태
  - we're &rArr; we are
- 여러 개의 단어가 붙어야 의미가 있는 경우
  - New york, rock'n'roll

# 한국어의 경우
- 한국어의 경우 토큰화가 복잡함
- 띄어쓰기가 잘 지켜지지 않고 띄어쓰기가 제대로 되었더라도 한 어절은 하나 이상의 의미 단위들이 있을 수 있음
- 형태소(morpheme): 뜻을 가진 가장 작은 말의 단위
  - 자립형태소: 명사, 대명사, 부사 등
  - 의존형태소: 다른 형태소와 결합하여 사용되는 형태소. 접사, 어미, 조사 등
- 단어보다 작은 단위(subword)로 토큰화가 필요함 을 알 수 있음.

## Subword Tokenization
- 만약 학습데이터에서 보지 못했떤 새로운 단어가 나타난다면
  - 학습데이터: low, new, newer
  - 테스트데이터: lower
  - -er, -est 등고 ㅏ같은 형태소를 분리할 수 있으면 좋을 것이다.
- Subword tokenization algorithms
  - Byte-Pair Encoding(BPE)
  - WordPiece
  - Unigram Language Modeling
- 두가지 구성요소
  - Token learner: 말뭉치에서 vocabulary(token 들의 집합)을 만들어냄
  - Token segmenter: 새로운 문장을 토큰화 함

### Byte-Pair Encoding (BPE)
- Vocabulary 를 단일 문자들의 집합으로 초기화 한다.
- 다음을 반복한다
  - 말뭉치에서 연속적으로 가장 많이 발생하는 두 개의 기호들(vocabulary내의 원소들)을 찾는다.
  - 두 기호들을 병합하고 새로운 기호로 vocabulary에 추가한다.
  - 말뭉치에서 그 두 기호들을 병합된 기호로 모두 교체한다.
- 위 과정을 k번의 병합이 일어날 때 까지 반복한다.

    ```c
    function BYTE-PAIR ENCODING(strings C, number of merges k) returns vocab V
        V ← all unique characters in C      # initial set of tokens is characters
        for i = 1 to k do                   # merge tokens til k times
            t_{L}, t_{R}              ← Most frequent pair of adjacent tokens in C
            t_{NEW} ← t_{L} + t_{R}         # make new token by concatenating
            V ← V + t_{NEW}                 # update the vocabulary
            Replace each occurrence of t_{L}, t_{R} in C with t_{NEW} # and update the corpus
        return V
    ```
- 기호병합은 단어안에서만 이루어진다. 이것을 위해서 단어끝을 나타내는 특수기호 '_'을 단어 뒤에 추가한다. 그리고 각 단어를 문자단위로 쪼갠다.
- 예제 말뭉치
  - low, low, low, low, low, lowest, lowest, newer, newer, newer, newer, newer, newer, wider, wider, wider, new, new
  - corpus
    - 5 l o w _
    - 2 l o w e s t_
    - 5 n e w e r _
    - 5 w i d e r _
    - 5 n e w _
  - vocabulary
    - _, d, e, i, l, n, o, r, s, t, w
- `e r`을 `er`로 병합
  - corpus
    - 5 l o w _
    - 2 l o w e s t_
    - 6 n e w er _
    - 3 w i d er _
    - 5 n e w _
  - vocabulary
    - _, d, e, i, l, n, o, r, s, t, w, er
- `er _`을 `er_`로 병합
  - corpus
    - 5 l o w _
    - 2 l o w e s t_
    - 6 n e w er_
    - 3 w i d er_
    - 5 n e w _
  - vocabulary
    - _, d, e, i, l, n, o, r, s, t, w, er, er\_
- `n e`을 `ne`로 병합
  - corpus
    - 5 l o w _
    - 2 l o w e s t_
    - 6 ne w er_
    - 3 w i d er_
    - 5 ne w _
  - vocabulary
    - _, d, e, i, l, n, o, r, s, t, w, er, er\_, ne
- 다음과 같이 병합들이 일어남
  - 
    |Merge |Current Vocabulary|
    |---|---|
    |(ne, w) |\_, d, e, i, l, n, o, r, s, t, w, er, er\_ , ne, new|
    |(l, o) |\_, d, e, i, l, n, o, r, s, t, w, er, er\_ , ne, new, lo|
    |(lo, w) |\_, d, e, i, l, n, o, r, s, t, w, er, er\_ , ne, new, lo, low|
    |(new, er ) |\_, d, e, i, l, n, o, r, s, t, w, er, er\_ , ne, new, lo, low, newer\_|
    |(low, ) |\_, d, e, i, l, n, o, r, s, t, w, er, er\_ , ne, new, lo, low, newer\_ , low\_|

- Token segmenter
  - 새로운 단어가 주어졌을 때 어떻게 토큰화할 것인지?
  - Greedy 한 적용: 병합을 학습한 순서대로 적용 ("e r" &rarr; "er")
  - 자주 나타나는 단어는 하나의 토큰으로 병합됨
  - 드문 단어는 subword 토큰들로 분할됨
- 하나의 단어 "n e w e r _" 은 하나의 토큰 "newer\_"로 토큰화 됨
- 하나의 단어 "l o w e r _"은 두 개의 토큰들 "low er\_"로 토큰화 됨
  - "low" "er\_"로 붙는 경우는 없었기 때문에.

### Wordpiece
- 기호들의 쌍을 찾을 때 빈도수 대신에 likelihood를 최대화시키는 쌍을 찾는다.
  - `likelihood`: 문장을 확률로 해석하는 것. (그 문장이 나타날 확률)
  ![wordpiece-likelihood.png](\assets\images\wordpiece-likelihood.png){: .align-center .img-80}

### Unigram
- 확률 모델(언어모델)을 사용함
- 학습데이터내의 문장을 관측(observed) 확률변수로 정의함
- Tokenization을 잠재(latent) 확률 변수로 정의함
  - 연속적인 (sequantial)
- 데이터의 주변 우도(marginal likelihood)를 최대화시키는 tokenization을 구함
  - EM (expectation maximization) 을 사용
  - Maximization step 에서 Viterbi 알고리즘을 사용(wordpiece는 greedy likelihood를 향상)


## 단어 정규화
- 단어들을 정규화된 형식으로 표현
  - U.S.A. or USA or US
  - uhhuh or uh-huh
  - Feb or fed
  - am, is be, are
- 검색 엔진에서 문서들을 추출할 때 유용할 수 있다.
  - 검색어: "USA", 문서: 단어 "US"만을 포함
  - **inverted index**(역색인)
    - 문서에 나타나는 모든 단어를 분석해서, 단어들 마다 이 단어를 포함하고 있는 문서들의 index를 만든다. 

## Case Folding
- 모든 문자들을 소문자화함
  - 일반화를 위해서 유용: 학습데이터와 테스트데이터 사이의 불일치 문제에 도움
  - 정보검색, 음성인식 등에서 유용
  - 감성분석 등 문서분류 문제에서는 오히려 대소문자 구분이 유용할 수 있음(국가이름 "US" vs 대명사 "us")

## Lemmatization
- 어근을 사용해서 표현
  - am, are, is, &rarr; be
- He is reading detective stories &rarr; He be read detective story

## 최근경향
- 단어 정규화가 필요한 근본적인 이유
  - 단어들 사이의 유사성을 이해해야하기 때문
  - 단어정규화 작업을 같은 의미를 가진 여러 형태의 단어들을 하나의 단어로 대응시키는 것으로 이해
- 단어를 vocabulary 로 정의된 공간(고차원 희소벡터)이 아닌 저차원 밀집 벡터로 대응시킬 수 있다면?
  - car &rarr; [0.13, 0.52, 0.01]
  - cars &rarr; [0.15, 0.49, 0.02]
  - 단어 임베딩을 사용해서 단어를 표현하게 되면 단어 정규화의 필요성이 줄어들게 됨.


# Appendix
## Reference
> Speech and Language Processing - 2.4 Text Normalization : <https://web.stanford.edu/~jurafsky/slp3/ed3book_dec302020.pdf>