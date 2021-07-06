---
date: 2021-07-06 00:38
title: "SQL Analysis - SQL & DB"
categories: DevCourse2 SQL MathJax
tags: DevCourse2 SQL MathJax
## 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# SQL의 중요성
데이터 관련 3개의 직군
- 데이터 엔지니어
    - Python, Java/Scalar
    - SQL, DB
    - ETL/ELT(airflow, DBT)
    - Spark, Hadoop
- 데이터 분석가
    - SQL, Domain Knowledge
    - Statistics (AB Test Analysis)
- 데이터 사이언티스트
    - ML
    - SQL, Python
    - Statistics  

# 배움
아는 것과 모르는 것을 분명히 이해하는 지.  

# Relational DB
- **구조화**된 데이터를 저장하는 데 사용됨.  
- 구조화된 데이터를 저장하고 질의할 수 있도록 해주는 스토리지
    - 엑셀 스프레트 시트 형태의 테이블로 데이터를 정의하고 저장
        - 데이터에는 컬럼과 레코드가 존재
- 관계형 DB를 조작하는 프로그래밍 언어가 SQL
    - 테이블 정의(Table Schema)를 위한 DDL (Data Definition Language)
    - 테이블 데이터 조작/질의를 위한 DML (Data Manipulation Language)  

## 대표적 RDB
- 프로덕션 DB: MySQL, PostgreSQL, Oracle, ...
    - OLTP (OnLine Transaction Processing)
    - 빠른 속도에 집중.
    - 서비스에 필요한 정보 저장
- 데이터 웨어하우스: Redshift, Snowflake, BigQuery, Hive, ...
    - OLAP (OnLine Analytical Processing)
    - 처리 데이터 크기에 집중
    - 데이터 분석 혹은 모델 빌딩등을 위한 데이터 저장
        - 보통 프로덕션 DB 를 복사해서 DW 에 저장.  

## RDB의 구조
- 2단계로 구성
    - 가장 밑단에는 테이블들이 존재(엑셀 시트)
    - 테이블들은 DB(혹은 Schema)라는 폴더 밑으로 구성(엑셀 파일)
- 테이블의 구조(테이블 Schema라고 부르기도 함)
    - 테이블은 레코드들로 구성(행)
    - 레코드는 하나 이상의 필드(컬럼)로 구성(열)
    - 필드(컬럼)는 이름과 타입과 속성(primary key)으로 구성  

# SQL이란
Structured Query Language  

- 두 종류의 언어로 구성
    - DDL (data definition language)
        - 테이블 구조를 정의
    - DML (data manipulation language)
        - 테이블에서 원하는 레코드들을 읽어오는 질의 언어
        - 테이블에 레코드를 추가/삭제/갱신하는데 사용하는 언어
- 모든 대용량 DW 는 SQL 기반
    - Redshift, Snowflake, BigQuery, Hive
- Spark 나 Hadoop도 예외는 아님
    - SparkSQL과 Hive라는 SQL 언어가 지원됨  

## SQL의 단점
- 구조화된 데이터를 다루는데 최적화가 되어있음
    - 정규표현식을 통해 비구조화된 데이터를 어느 정도 다루는 것은 가능하나 제약이 심함.
    - 많은 RDB 들이 플랫한 구조만 지원함 (no nested, like json)
        - 구글 BigQuery는 nested structure를 지원함
    - 비구조화된 데이터를 다루는데 Spark, Hadoop과 같은 분산 컴퓨팅 환경이 필요해짐
        - 즉, SQL 만으로는 비구조화 데이터를 처리하지 못함
- RDB마다 SQL 문법이 조금씩 다름.  

## 데이터 모델링
Star Schema, Denormalized Schema  

### Star schema
- Production DB용 RDB에서는 보통 Star schema를 사용해 데이터를 저장
- 데이터를 논리적 단위로 나눠 저장하고 필요시 조인
- 스토리지 낭비가 덜함
- 업데이트가 쉬움
- 레코드를 보려면 조인을 해야함.  

### Denormalized Schema
- DW 에서 사용
    - 단위 테이블로 나눠 저장하지 않음
    - 별도의 조인이 필요 없음
- 스토리지를 더 사용하지만 조인이 필요없기에 빠른 계산 가능  
- DW에서는 특정 레코드를 업데이트 하는 경우가 거의 없음.  


# 데이터 웨어하우스란
회사에 필요한 모든 데이터를 저장
- SQL 기반의 RDB
    - Production DB와는 별도이어야 함
        - OLAP vs. OLTP
    - AWS의 Redshift, Google Cloud의 Big Query, Snowflake 등이 대표적
        - 고정비용 옵션 vs. 가변비용 옵션
- DW 는 내부 직원을 위한 DB
    - 처리 속도가 아닌 데이터의 크기가 중요
- ETL (Extract-Transform-Load) 혹은 데이터 파이프라인
    - 외부에 존재하는 데이터를 읽어다가 DW로 저장해주는 코드들이 필요해지는데 이를 ETL 혹은 데이터 파이프라인이라고 부름  

## 데이터 인프라
- 데이터 엔지니어가 관리
    - 한 단계 더 발전하면 Spark와 같은 대용량 분산처리 시스템이 일부로 추가됨.  

![data-infra](/assets/images/data-infra.png){: .align-center .img-70}  

## 데이터 순환 구조
![data-cycle](/assets/images/data-cycle.png){: .align-center .img-70}  

# Cloud AWS
- 클라우드의 정의
    - 컴퓨팅 자원을 네트워크를 통해 서비스 형태로 사용하는 것
    - 키워드: No provisioning, Pay as you go
    - 자원을 필요한만큼 실시간으로 할당하여 사용한만큼 지불
            - 탄력적으로 필요한만큼의 자원을 유지하는 것이 중요.  

## Cloud Computing이 없다면?
- 서버/네트워크/스토리지 구매와 설정 등을 직접해야
- 데이터 센터 공간을 직접 확보(Co-location)
    - 확장이 필요한 경우 공간을 먼저 더 확보해야함
- 그 공간에 서버를 구매하여 설치하고 네트워크 설정
    - 서버 구매와 설치 두세달 소요
- Peak Time을 기준으로 Capacity Planning을 해야함.
    - 노는 자원들이 존재
- 직접 운영비용 vs. 클라우드 비용
    - 기회비용(기다리지 않아도 됨)  

## Cloud Computing의 장점
- 초기 투자비용이 줄어듬
    - CAPEX (Capital Expenditure) Vs. OPEX(Operating Expense)
- 리소스 준비를 위한 대기시간 대폭 감소
    - Shorter time to Market
- 노는 리소스 제거로 비용 감소
- 글로벌 확장 용이
- 소프트웨어 개발 시간 단축
    - SaaS 이용(Managed Service)  

## AWS
- 가장 큰 클라우드 컴퓨팅 서비스 업체
- 아마존 상품데이터를 API로 제공하면서 시작
    - 최근 ML/AI 서비스 제공 시작
- 사용 고객
    - Netflix, Zynga 등
    - 많은 국내 업체들
- 다양한 종류의 소프트웨어/플랫폼 서비스 제공  

### EC2 - Elastic Compute Cloud
- AWS의 서버 호스팅 서비스
    - 서버를 론치하고 계정 생성해서 로그인 가능
    - 가상 서버들이라 전용서버에 비해 성능 저하
    - Bare-metal 서버도 제공
- 다양한 종류의 서버 타입 제공
- 세 종류의 구매 옵션
    - On-Demand: 시간당 비용 지불
    - Reserved: 1년 혹은 3년간 사용 보장하고 30 ~ 40% 할인
    - Spot Instance: 일종의 경매방식으로 놀고 있는 리소스들 저렴한 비용으로 사용할 수 있음
        - 경매가가 높은 사람에게 서버가 넘어가기 때문에 서버가 항상 살아있다고 가정 불가  

### S3 - Simple Storage Service
- 아마존이 제공하는 대용량 클라우드 스토리지 서비스
- 데이터 저장관리를 위해 계층적 구조를 제공
- 글로벌 네임스페이스를 제공하기 때문에 탑레벨 디렉토리 이름 선정에 주의
- S3에서는 디렉토리를 버킷(Bucket)이라고 부름
- 버킷이나 파일별로 액세스 컨트롤 가능  

### 기타 중요 서비스 - DB Service
- RDS(Relational DB Service)
    - MySQL, PostgreSQL, Aurora
    - Oracle, MS SQL Server
- DynamoDB
- Redshift
- ElasticCache
- Neptune(Graph DB)
- Elastic Search
- MongoDB  

### 기타 중요 서비스 - AI & ML Services
- SageMaker
    - DL & ML E2E framework
    - API
    - Deploy 까지
- Lex
    - Conversational Interface (Chatbot Service)
- Polly
    - Text to Speech Engine
- Rekognition
    - Image Recognition Service  

### 기타 중요 서비스
- Amazon Alexa
    - Voice Bot Platform
- Amazon Connect
    - Contact Center Solution
    - 콜센터 구현
- Lambda
    - Event-driven, serverless computing enging
    - 서비스 구현을 위해 EC2 론치 불필요
    - Google Cloud에서는 Cloud Function
    - Azure에는 Azure Function  

# Redshift
- Scalable SQL 엔진
    - 2 PB 까지 지원
    - Still OLAP (DW)
        - 응답속도가 빠르지 않기 때문에 Production DB로 사용 불가
    - Columnar Storage
        - 레코드별로 저장하는 것이 아니라 컬럼별로 저장하는 형태
        - 컬럼별 압축이 가능
        - 컬럼을 추가하거나 삭제하는 것이 아주 빠름
    - 벌크 업데이트 지원
        - 레코드가 들어있는 파일을 S3(Web storage)로 복사 후 COPY 커맨드로 Redshift로 일괄 복사
    - 고정 용량/비용 SQL 엔진
        - vs. Snowflake vs. BigQuery (가변)
    - 다른 DW 처럼 primary key uniqueness를 보장하지 않음
        - 보장하려면, 레코드가 추가될 때마다, primary key로 존재하는 필드에 중복이 있는지 체크를 해야함
            - 속도가 느려짐
        - Production DB는 보장함.
- PostgreSQL 8.x와 SQL이 호환
    - 하지만, PostgreSQL 8.x의 모든 기능을 지원하지는 않음
        - 예를 들어 text 타입이 존재하지 않음
    - PostgreSQL 8.x을 지원하는 툴이나 라이브러리로 액세스 가능
        - JDBC/ODBC
    - SQL이 메인언어
        - 테이블 디자인이 아주 중요

## Redshift Schema (폴더) 구성
## Redshift 액세스 방법
- Google Colab
- PostgreSQL 8.x와 호환되는 모든 툴과 프로그래밍 언어를 통해 접근 가능
    - SQL Workbench(Mac, Windows), Postico(Mac)
    - Python이라면 psycopg2 모듈
    - 시각화/대시보트 툴이라면 Looker, Tableau, Power BI, Superset 등에서 연결가능
