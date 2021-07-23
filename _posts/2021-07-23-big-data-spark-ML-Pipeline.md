---
date: 2021-07-23 15:34
title: "Big data - ML Pipeline과 Tuning 소개와 실습"
categories: DevCourse2 Spark BigData SparkMLlib
tags: DevCourse2 Spark BigData SparkMLlib
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Spark MLlib 모델 튜닝 소개
## Spark MLlib 모델 튜닝
- 최적의 하이퍼 파라미터 선택
  - 최적의 모델 혹은 모델의 파라미터를 찾는 것이 아주 중요
  - 하나씩 테스트해보는 것 vs. 다수를 동시 테스트 하는 것
  - 모델 선택의 중요한 부분은 테스트 방법
    - Cross Validation 과 Hold Out 테스트 방법을 지원
  - 보통 ML Pipeline과 같이 사용

## 모델 테스트 방법: 크게 2가지가 존재
- Cross Validation: `CrossValidator`
- Train-validation split: `TrainValidationSplit`

### Train-Validation Split
- Hold out 테스트라고 하기도

### Cross Validation
- K-fold 
- 홀드아웃보다 안정적
  - 오버피팅 문제 감소


### `CrossValidator`, `TrainValidationSplit`
- `TrainValidationSplit`: 홀드아웃 기반
- `CrossValidator`: Kfold 기반
- 다음과 같은 입력을 기반으로 가장 좋은 파라미터를 찾아줌
  - `Estimator` (머신러닝 모델 혹은 ML Pipeline)
  - `Evaluator` (머신러닝 모델의 성능을 나타내는 지표)
  - `Parameter` (훈련 반복 회수 등의 하이퍼 파라미터)
    - `ParamGridBuilder` 를 이용해 `ParamGrid` 타입의 변수 생성
  - 최종적으로 가장 결과가 좋은 모델 리턴

#### `Evaluator`
- `evaluate` 함수가 제공됨
  - 인자로 테스트 셋의 결과가 들어있는 데이터프레f임과 파라미터가 제공
- 머신러닝 알고리즘에 따라 다양한 Evaluator 가 제공됨
  - RegressionEvaluator
  - BinaryClassificationEvaluator
    - AUC
  - MulticlassClassificationEvaluator
  - MultilabelClassificationEvaluator
  - RankingEvaluator
![evaluator](/assets/images/evaluator.png){: .align-center .img-80}

## 모델 선택시 입력
- Estimator
  - 머신러닝 알고리즘이나 모델 빌딩 파이프라인
- Evaluator
  - 머신러닝 모델 성능 측정에 사용되는 지표
- 모델의 param map
  - ParamGrid 라고 불리기도 하는데 모델 테스트시 고려해야하는 가능한 러닝 관련 파라미터들로 주로 트리 관련 알고리즘에서 중요
  - 테스트되는 파라미터의 예로는 depth, iterations

## Spark MLlib 머신러닝 모델 빌딩 전체 프로세스
![MLlib-process](/assets/images/MLlib-process.png){: .align-center .img-80}  

***  

![MLlib-process-2](/assets/images/MLlib-process-2.png){: .align-center .img-80}

# 타이타닉 승객 생존 예측 모델을 ML Pipeline과 Tuning으로 빌딩
## 타이타닉 승객 예측 분류기 개발 방향
- ML Pipeline을 사용하여 모델 빌딩
- 다양한 Transformer 사용
  - Imputer, StringIndexer, VectorAssembler
  - MinMaxScaler 를 적용하여 피쳐 값을 0과 1사이로 스케일
- GBT Classifier 와 LogisticRegression을 머신러닝 알고리즘으로 사용
  - Gradient Boosted Tree Classifier
    - 의사결정 트리의 머신러닝 알고리즘
    - Regression 과 Classification 모두에 사용 가능
- CrossValidatoin 을 사용하여 모델 파라미터 선택
  - BinaryClassificationEvaluator 를 Evaluator 로 사용
  - ParamGridBuiler 를 사용하여 ParamGrid 생성
  - 뒤에 설명할 ML Pipeline을 인자로 지정
    - ML Pipeline 을 만들 때 머신러닝 알고리즘을 마지막에 지정해야 함

### MinMaxScaler 사용
- 기본적으로 VectorAssembler 로 벡터로 변환된 피쳐컬럼에 적용
    ```py
    from pyspark.ml.feature import MinMaxScaler
    age_scaler = MinMaxScaler(inputCol='features',outputCol='features_scaled')
    age_scaler_model = age_scaler.fit(data_vec)
    data_vec = age_scaler_model.transform(data_vec)

    data_vec.select("features","features_scaled").show()
    ```
- 이 경우 AUC 가 0.03 정도 올라감
    ```py
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], 'r--')
    plt.plot(model.summary.roc.select("FPR").collect(),
        model.summary.roc.select("TPR").collect())
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()
    ```

## ML Pipeline 사용 절차
- 트레이닝 셋에 수행해야 하는 feature transformer 들을 생성
- 사용하고자 하는 머신러닝 모델 알고리즘 (Estimator)을 생성
- 이 것들을 순서대로 파이썬 리스트에 추가
  - 머신러닝 알고리즘이 마지막으로 추가되어야 함
- 이 파이썬 리스트를 인자로 Pipeline 개체를 생성
- 이 Pipeline 개체를 이용해 모델 빌딩: 2가지 방법 존재
  1. 이 Pipeline 의 fit 함수를 호출하면서 트레이닝 셋 데이터프레임을 지정
  2. 이 Pipeline 을 ML Tuning 개체로 지정해서 여러 하이퍼 파라미터를 테스트해보고 가장 결과가 좋은 모델을 선택

### ML Pipeline 사용 예
- 필요한 Transformer 와 Estimator 들을 만들고 순서대로 리스트에 추가
    ```py
    from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler, MinMaxScaler
    from pyspark.ml.classification import LogisticRegression
    # Gender
    stringIndexer = StringIndexer(inputCol = "Gender", outputCol = 'GenderIndexed')
    # Age
    imputer = Imputer(strategy='mean', inputCols=['Age'], outputCols=['AgeImputed'])
    # Vectorize
    inputCols = ['Pclass', 'SibSp', 'Parch', 'Fare', 'AgeImputed', 'GenderIndexed']
    assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
    # MinMaxScaler
    minmax_scaler = MinMaxScaler(inputCol="features", outputCol="features_scaled")

    stages = [stringIndexer, imputer, assembler, minmax_scaler]
    algo = LogisticRegression(featuresCol="features_scaled", labelCol="Survived")
    lr_stages = stages + [algo]
    ```

- 앞서 만든 리스트를 Pipeline 의 인자로 지정

    ```py
    from pyspark.ml import Pipeline
    pipeline = Pipeline(stages = lr_stages)

    df = data.select(['Survived', 'Pclass', 'Gender', 'Age', 'SibSp', 'Parch', 'Fare'])
    train, test = df.randomSplit([0.7, 0.3])
    lr_model = pipeline.fit(train)
    lr_cv_predictions = lr_model.transform(test)
    evaluator.evaluate(lr_cv_predictions)
    ```

## ML Tuning 사용 절차
- 테스트 하고 싶은 머신러닝 알고리즘 개체 생성 (혹은 ML Pipeline)
- ParamGrid 를 만들어 테스트하고 싶은 하이퍼 파라미터 지정
- CrossValidator 혹은 TrainValidationSplit 생성
- fit 함수 호출해서 최선의 모델 선택

## ML Tuning 사용 예
- ParamGrid 와 CrossValidator 생성
    ```py
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    evaluator = BinaryClassificationEvaluator(labelCol='Survived', metricName='areaUnderROC')

    from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    paramGrid = (ParamGridBuilder()
                .addGrid(algo.maxIter, [1, 5, 10])
                .build())
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator, 
        numFolds=5
    )
    ```

- CrossValidator 실행하여 최선의 모델 선택하고 테스트 셋으로 검증

    ```py
    train, test = df.randomSplit([0.7, 0.3])
    # Run cross validations.
    cvModel = cv.fit(train)
    # 테스트셋을 입력으로 주고 결과를 받아 분석
    lr_cv_predictions = cvModel.transform(test)
    evaluator.evaluate(lr_cv_predictions)
    ```

- 어떤 하이퍼 파라미터 조합의 최선의 결과를 냈는 지 알고 싶다면
  - ML Tuning 의 `getEstimatorParamMaps`/`getEvaluator` 의 조합으로 파악
  - Logistic Regression 의 경우 아래와 같은 결과 도출
    ```py
    import pandas as pd
    params = [{p.name: v for p, v in m.items()} for m in cvModel.getEstimatorParamMaps()]
    pd.DataFrame.from_dict([
        {cvModel.getEvaluator().getMetricName(): metric, **ps} 
        for ps, metric in zip(params, cvModel.avgMetrics)
    ])
    # areaUnderROC	maxIter
    # 0	0.832558	1
    # 1	0.846819	5
    # 2	0.845913	10
    ```

# Spark ML 모델을 API로 서빙 - PMML (범용 머신러닝 모델 파일포맷)
- 다양한 머신러닝 개발 플랫폼 들이 넘쳐남
- 통용되는 머신러닝 파일포맷이 있다면 어떨까
- 그래서 나온 모듈 혹은 파일포맷등이 있음
  - PMML 과 MLeap 이 대표적
  - 머신러닝 모델 서빙환경의 통일이 가능
    - 실상은 이런 공통 파일포맷이 지원해주는 기능이 미약해서 복잡한 모델의 경우에는 지원불가

## PMML: Predictive Model Markup Language
- ML 모델을 마크업 언어로 표현해주는 XML 언어
  - 간단한 입력 데이터 전처리와 후처리도 지원. 하지만 아직도 제약사항이 많음
  - Java 기반
    - <https://github.com/jpmml/jpmml-evaluator>
    - 많은 회사들이 모델 실행을 위해서 자바로 PMML 엔진을 구현
  - PySpark 에서는 pyspark2pmml를 사용
    - 하지만 내부적으로는 jpmml-sparkml 이라는 자바 jar 파일을 사용
    - 너무 복잡하고 버전 의존도도 심함

## 전체적인 절차
1. ML Pipeline 을 PMML 파일로 저장
   - 이를 위해 pyspark2pmml 파이썬 모듈을 설치
   - pyspark2pmml.PMMLBuilder 를 이용하여 ML Pipeline을 PMML 파일로 저장
2. PMML 파일을 기반으로 모델 예측 API 로 론치
   - Openscoring Framework
   - AWS SageMaker
   - Flask + PyPMML
3. 이 API 로 승객정보를 보내고 예측 결과를 받는 클라이언트 코드 작성

## 머신러닝 모델을 PMML 파일로 저장하는 예제
- cvModel 은 머신러닝 모델이나 ML Pipeline
- train_fr 은 트레이닝 셋 데이터프레임

    ```py
    from pyspark2pmml import PMMLBuilder
    pmmlBuilder = PMMLBuilder(spark.sparkContext, train_fr, cvModel)
    pmmlBuilder.buildFile("Titanic.pmml")
    ```

## PMML 파일을 PyPMML 로 로딩하고 호출하는 예제
- 로딩 예제(Model.load)
    ```py
    from pypmml import Model
    model = Model.load('single_iris_dectree.pmml')
    ```

- 예측 예제(predict)
    ```py
    model.predict({'sepal_length':5.1, 'sepal_width':3.5, 'petal_length':1.4, 'petal_length':0.2})
    ```

# Appendix
## Reference
> notebook: <https://github.com/learn-programmers/programmers_kdt_II/blob/main/11%EC%A3%BC%EC%B0%A8_PySpark_%EA%B8%B0%EB%B3%B8_5%EC%9D%BC%EC%B0%A8_1.ipynb>