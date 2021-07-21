---
date: 2021-07-21 22:25
title: "Big data - Spark MLlib 소개와 머신러닝 모델 빌딩"
categories: DevCourse2 Spark BigData SparkMLlib
tags: DevCourse2 Spark BigData SparkMLlib
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Spark MLlib 소개
- 머신러닝 관련 다양한 알고리즘, 유틸리티로 구성된 라이브러리
  - Classification, Regression, Clustering, Collaborative Filtering, Dimensionality Reduction, ...
  - 아직 딥러닝 지원은 미약
- 여기에는 RDD 기반과 데이터프레임 기반의 두 버전이 존재
  - spark.mllib vs. spark.ml
    - spark.mllib가 RDD 기반이고 spark.ml은 데이터프레임 기반
    - spark.mllib는 RDD위에서 동작하는 이전 라이브러리로 더 이상 업데이트가 안됨
  - 항상 spark.ml 을 사용할 것
    - import pyspark.ml

## Spark MLlib의 장점
- 원스톱 ML Framework
  - 데이터프레임과 SparkSQL 등을 이용해 전처리
  - Spark MLlib 을 이용해 모델 빌딩
  - ML Pipeline을 통해 모델 빌딩 자동화
  - MLflow로 모델 관리하고 서빙
- 대용량 데이터도 처리 가능
- 데이터가 작은 경우 굳이 사용할 필요 없음

## ML flow
- 모델의 관리와 서빙을 위한 Ops(Operations) 관련 기능도 제공
- MLflow
  - 모델 개발과 테스트와 관리와 서빙까지 제공해주는 end-to-end framework
  - MLflow는 파이썬, 자바, R, API를 지원
  - MLflow는 트래킹, 모델, 프로젝트를 지원

## Spark MLlib 제공 알고리즘
- Classification
  - Logistic Regression, Decision tree, Random Forest, Gradient-boosted tree, ...
- Regression
  - Linear Regression, Decision tree, Random Forest, Gradient-boosted tree, ...
- Clustering
  - K-means, LDA(Latent Dirichlet Allocation), GMN(Gaussian Mixture Model), ...
- Collaborative Filtering
  - 명시적인 피드백과 암묵적인 피드백 기반
  - 명시적인 피드백의 예) 리뷰 평점
  - 암묵적인 피드백의 예) 클릭, 구매 등


# 머신러닝 모델링 실습
## Spark MLlib 기반 모델 빌딩의 기본구조
- 여느 라이브러리를 사용한 모델 빌딩과 크게 다르지 않음
  - 트레이닝셋 전처리
  - 모델 빌딩
  - 모델 검증(confusion matrix)
- Scikit-learn과 비교했을 때 장점
  - 데이터의 크기
  - sklearn은 하나의 컴퓨터에서 돌아가는 모델 빌딩
  - Spark MLlib은 여러 서버 윙에서 모델 빌딩
- 트레이닝 셋의 크기가 크면 전처리와 모델 빌딩에 있어 Spark이 큰 장점
- Spark은 ML 파이프라인을 통해 모델 개발의 반복을 쉽게 해줌


## 보스턴 주택가격 예측 모델 만들기: Regression
- 1970 년대 미국 인구조사 서비스에서 보스턴 지역의 주택가격 데이터를 수집한 데이터를 기반으로 모델 빌딩
- 개별 주택가격의 예측이 아니라 지역별 중간 주택가격 예측
- Regression 알고리즘 사용

### Training set
- 총 506개의 레코드로 구성되며 13개의 피쳐와 label(주택가격)으로 구성
  - 506개 동네의 주택 중간값 데이터임 (개별 주택이 아님)
  - 14번째 필드가 바로 예측해야 하는 중간 주택가격

    |Variable	|Definition|
    |---|---|
    |crim| per capita crime rate by town.|
    |zn| proportion of residential land zoned for lots over 25,000 sq.ft.|
    |indus| proportion of non-retail business acres per town.|
    |chas| Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).|
    |nox| nitrogen oxides concentration (parts per 10 million). (오염정도)|
    |rm| average number of rooms per dwelling.|
    |age| proportion of owner-occupied units built prior to 1940.|
    |dis| weighted mean of distances to five Boston employment centres.|
    |rad| index of accessibility to radial highways.|
    |tax| full-value property-tax rate per 10,000 dollars.|
    |ptratio| pupil-teacher ratio by town.|
    |black| $$1000(Bk−0.63)^2$$ where $$Bk$$ is the proportion of blacks by town.|
    |lstat| lower status of the population (percent).|
    |medv(label)| median value of owner-occupied homes in 1000 dollars(천불단위)|

### 실습
<https://github.com/learn-programmers/programmers_kdt_II/blob/main/11%EC%A3%BC%EC%B0%A8_PySpark_%EA%B8%B0%EB%B3%B8_4%EC%9D%BC%EC%B0%A8_1.ipynb>  



## 타이타닉 승객 생존 예측 모델 만들기: Classification
- AUC의 값이 중요한 성능 지표가 됨
  - True Positive Rate과 False Positive Rate

### Training set
- columns
  
    |Variable	|Definition|	Key|
    |---|---|---|
    |survival 	|Survival |	0 = No, 1 = Yes|
    |pclass |	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd|
    |sex 	|Sex 	|
    |Age 	|Age in years 	|
    |sibsp 	|# of siblings / spouses aboard the Titanic 	|
    |parch 	|# of parents / children aboard the Titanic 	|
    |ticket 	|Ticket number 	|
    |fare 	|Passenger fare 	|
    |cabin 	|Cabin number 	|
    |embarked 	|Port of Embarkation 	|C = Cherbourg, Q = Queenstown, S = Southampton|

### 실습
<https://github.com/learn-programmers/programmers_kdt_II/blob/main/11%EC%A3%BC%EC%B0%A8_PySpark_%EA%B8%B0%EB%B3%B8_4%EC%9D%BC%EC%B0%A8_2.ipynb>  

# Spark MLlib 피쳐변환
## 피쳐추출과 변환
- 피여 값들을 모델 훈련에 적합한 형태로 바꾸는 것을 지칭
- 크게 두 가지가 존재: `Feature Extractor`와 `Feature Transformer`

### `Feature Transformer`가 하는 일
  - 먼저 피쳐 값들은 숫자 필드이어야 함
    - 텍스트 필드(카테고리 값들)를 숫자 필드로 변환해야 함
  - 숫자 필드 값의 범위 표준화
    - 숫자 필드라고 해도 가능한 값의 범위를 특정 범위(0부터 1)로 변환해야 함
    - 이를 피쳐 스케일링(Feature Scaling) 혹은 정규화(Normalization)
  - 비어있는 필드들의 값을 어떻게 채울 것인가?
    - Imputer

### `Feature Extractor`가 하는 일
- 기존 피쳐에서 새로운 피쳐를 추출
- TF-IDF, Word2Vec, ...
  - 많은 경우 텍스트 데이터를 어떤 형태로 인코딩하는 것이 여기에 해당함

### `Feature Transformer` -  `StringIndexer`: 텍스트 카테고리를 숫자로 변환
[docs](https://spark.apache.org/docs/latest/ml-features#stringindexer)
> StringIndexer encodes a string column of labels to a column of label indices.  

| id | category | categoryIndex|
|----|----------|---------------|
| 0  | a        | 0.0|
| 1  | b        | 2.0|
| 2  | c        | 1.0|
| 3  | a        | 0.0|
| 4  | a        | 0.0|
| 5  | c        | 1.0|

- Scikit-Learn은 `sklearn.preprocessing` 모듈 아래에 여러 인코더가 존재함.
  - `OneHotEncoder`, `LabelEncoder`, `OrdinalEncoder`, ...
- Spark MLlib의 경우 `pyspark.ml.feature` 모듈 밑에 두 개의 인코더 존재
  - `StringIndexer`, `OneHotEncoder`
  - 사용법은 `Indexer` 모델을 만들고 (fit), `Indexer` 모델로 데이터프레임을 transform

```py
from pyspark.ml.feature import StringIndexer

gender_indexer = StringIndexer(inputCol='Gender', outputCol='GenderIndexed')
gender_indexer_model = gender_indexer.fit(final_data)
final_data_with_transformed_gender_gender = gender_indexer_model.transform(final_data)
```

### `Feature Transformer` -  `Scaler`: 숫자 필드값의 범위 표준화
- 숫자 필드 값의 범위를 특정 범위(0부터 1까지)로 변환하는 것
- 피쳐 스케일링 혹은 정규화라 부름
- Scikit-Learn은 `sklearn.preprocessing` 모듈 아래 두 개의 스케일러 존재
  - `StandardScaler`, `MinMaxScaler`
- Spark MLlib의 경우 `pyspark.ml.feature` 모듈 밑에 두 개의 스케일러 존재
  - `StandardScaler`, `MinMaxScaler`
  - fit 하고 transform
- `StandardScaler`
  - 각 값에서 평균을 빼고 이를 표준편차로 나눔. 값의 분포가 정규분포를 따르는 경우 사용
- `MinMaxScaler`
  - 모든 값을 0과 1사이로 스케일. 각 값에서 최소값을 빼고 (최대값-최소값)으로 나눔

### `Feature Transformer` -  `Imputer`: 값이 없는 필드 채우기
- 값이 없는 레코드들이 존재하는 필드들의 경우 기본값을 정해서 채우는 것
- Scikit-Learn은 `sklearn.preprocessing` 모듈 아래 존재
  - `Imputer`
- Spark MLlib의 경우 `pyspark.ml.feature` 모듈 밑에 존재
  - `Imputer`
  - fit and transfrom

```py
from pyspark.ml.feature import Imputer
imputer = Imputer(strategey='mean', inputCols=['Age'],outputCols=['AgeImputed'])
imputer_model = imputer.fit(final_data)
final_data_age_transformed = imputer_model.transform(final_data)
```

# Spark ML Pipeline 소개
## 모델 빌딩과 관련된 흔한 문제들
- a. 트레이닝 셋의 관리가 안됨
- b. 모델 훈련 방법이 기록이 안됨
  - 어떤 트레이닝 셋을 사용했는지
  - 어떤 피쳐들을 사용했는 지
  - 하이퍼 파라미터는 무엇을 사용했는지
- c. 모델 훈련에 많은 시간 소요
  - 모델 훈련이 자동화가 안된 경우 매번 각 스텝들을 노트북 등에서 일일히 수행
  - 에러가 발생할 여지가 많음(특정 스텝을 까먹거나 조금 다른 방식 적용)

## ML Pipeline의 등장
- b와 c를 해결
- 자동화를 통해 에러 소지를 줄이고 반복을 빠르게 가능하게 해줌
- Load data &rArr; Extract features &rArr; Train model &rArr; Evaluate

## Spark MLlib 관련 개념정리
- ML Pipeline이란
  - 데이터 사이언티스트가 머신러닝 개발과 테스트를 쉽게 해주는 기능 (데이터 프레임 기반)
  - 머신러닝 알고리즘에 관계없이 일관된 형태의 API를 사용하여 모델링이 가능
  - ML 모델 개발과 테스트를 반복가능하게 해줌
    - Transformer와 Estimator로 구성됨
  - 4개의 요소로 구성
    - 데이터프레임
    - Transformer
    - Estimator(모델)
    - Parameter(하이퍼파라미터)

### ML Pipeline 구성요소 - 데이터프레임
- ML Pipeline에서는 데이터프레임이 기본 데이터 포맷
- csv, json, parquet, JDBC(관계형 DB) 
- ML Pipeline에서 다음 2가지의 새로운 데이터 소스 지원
  - 이미지 데이터 소스
    - jpg, png 등
  - LIBSVM 데이터 소스
    - label과 feature 두 개의 컬럼으로 구성되는 머신러닝 트레이닝 셋 포맷
    - feature 컬럼은 벡터 형태의 구조

### ML Pipeline 구성요소 - Transformer
- 입력 데이터프레임을 다른 데이터프레임으로 변환
  - 하나 이상의 새로운 컬럼을 추가 
- 2 종류의 `transformer` 가 존재. **transform** 이 메인 함수
  - `Feature Transformer`와 `Learning Model`
- `Feature Transformer`
  - 입력 데이터 프레임의 컬럼으로부터 새로운 컬럼을 만들어내 이를 추가한 새로운 데이터프레임을 출력으로 내줌. 보통 FE를 하는데 사용
  - `Imputer`, `StringIndexer`, `VectorAssembler`
- `Learning Model`
  - 머신러닝 모델에 해당
    - 피쳐 데이터프레임을 입력으로 받아 예측값이 새로운 컬럼으로 포함된 데이터프레임을 출력으로 내줌: prediction, probability

### ML Pipeline 구성요소 - Estimator
- 머신러닝 알고리즘에 해당. **fit** 이 메인 함수
  - 트레이닝 셋 데이터프레임을 입력으로 받아서 머신러닝 모델을 만들어냄
    - 입력: 데이터프레임
    - 출력: 머신러닝 모델
  - 예를 들어 `LogisticRegression` 은 `Estimator` 이고 `LogisticRegression.fit()` 를 호출하면 머신러닝 모델(`Transformer`)을 만들어냄.
- ML Pipeline 도 Estimator
- Estimator 는 저장과 읽기 함수 제공
  - 즉, 모델과 ML Pipeline을 저장했다가 나중에 다시 읽을 수 있음
    - save, load

### ML Pipeline 구성요소 - Parameter
- Transformer 와 Estimator 의 공통 API로 다양한 인자를 적용해줌
- 두 종류의 파라미터가 존재
  - `Param` (하나의 이름과 값)
  - `ParamMap` (`Param` 리스트)
- 파라미터의 예
  - `iteration` 지정을 위해 `setMatIter()` 를 사용
  - `ParamMap(lr.maxIter &rarr; 10)`
- 파라미터는 fit (Estimator) 혹은 transform (Transformer) 에 인자로 지정 가능.

### ML Pipeline - Summary
- 하나 이상의 Transformer 와 Estimator 가 연결된 모델링 Workflow
  - 입력은 데이터프레임
  - 출력은 머신러닝 모델
- ML Pipeline 그 자체도 Estimator
  - 따라서 ML Pipeline 의 실행은 fit 함수의 호출로 시작
  - 저장했따가 나중에 다시 로딩 하는 것이 가능 (Persistence)
- 한번 파이프라인을 만들면 반복 모델빌딩이 쉬워짐