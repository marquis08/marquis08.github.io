---
date: 2021-07-21 02:39
title: "Big data - SparkSQL을 이용한 데이터 분석"
categories: DevCourse2 Spark BigData
tags: DevCourse2 Spark BigData
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# 커리어 이야기
- 남과 비교하지 말고 앞으로 20-30년을 보기
- 하나를 하기로 했으면 적어도 6개월은 파고 들기
  - 너무 빨리 포기하지 않기
  - 뭔가 잘 안되면 서두르기 보다는 오히려 천천히 가기
- 공부를 위한 공부를 하기 보다는 일을 시작해보기
  - 어디건 일을 시작하고 발전해 나가기
  - 면접 실패를 감정적으로 받아들이지 않기

## 새로운 시작 - 처음 90일이 중요
- 자기 검열하지 말고 매니저의 스타일 파악하고 피드백 요청
- 과거의 상처 가지고 시작하지 않기
- 남과 비교하지 않기
- 열심히 일하되 너무 서두르지 않기

## 새로운 기술의 습득이 아닌 결과를 내는데 초점 맞추기
- 아주 나쁘지 않은 환경에 있다는 전제
- 자신이 맡은 일을 잘 하기 위해서 필요한 기술습득
  - 예를 들면 자동화하기 혹은 실행시간 단축하기
- 자신이 맡은 일의 성공/실패를 어떻게 결정하는지 생각
  - 매니저와의 소통이 중요
  - 성공/실패 지표에 대해서 생각
- 일을 그냥 하지 말고 항상 "왜" 이 일이 필요하고 큰 그림을 생각
  - 질문하기


# SQL 이란
## SQL 은 빅데이터 세상에서도 중요
- 구조화된 데이터를 다루는 한 SQL은 데이터 규모와 상관없이 쓰임
- 모든 대용량 데이터 웨어하우스는 SQL 기반
  - Redshift, Snowflake, BigQuery, Hive
- Spark도 예외는 아님
  - SparkSQL이 지원
- 데이터 분야에서 일하고자 하면 반드시 익혀야할 기본 기술

## 관계형 데이터 베이스
- 대표적인 관계형 데이터 베이스
  - MySQL, Postgres, Oracle, ...
  - Redshift, Snowflake, BigQuery, Hive, ...
- 관계형 데이터베이스는 2단계로 구성
  - 가장 밑단에는 테이블들이 존재 (테이블은 엑셀의 시트에 해당)
  - 테이블들은 데이터베이스라는 폴더 밑으로 구성
- 테이블의 구조 (스키마라도 부르기도함)
  - 테이블은 레코드들로 구성
  - 레코드는 하나 이상의 필드로 구성
  - 필드는 이름과 타입으로 구성

## 관계형 데이터베이스 예제 - 웹서비스 사용자/세션 정보
- 사용자 ID:
  - 보통 웹서비스에서는 등록된 사용자마다 유일한 ID를 부여 &rArr; 사용자 ID
- 세션 ID:
  - 사용자가 외부 링크 (보통 광고)를 타고 오거나 직접 방문해서 올 경우 세션을 생성
  - 즉 하나의 사용자 ID는 여러 개의 세션 ID를 가질 수 있음
  - 보통 세션의 경우 세션을 만들어낸 소스를 채널이란 이름으로 기록해둠
    - 마케팅 관련 기여도 분석을 위함
  - 또한 세션이 생긴 시간도 기록
- 이 정보를 기반으로 다양한 데이터 분석과 지표 설정이 가능
  - 마케팅 관련
  - 사용자 트래픽 관련
- 사용자 ID 100번: 총 3개의 세션(painted)을 갖는 예제
  - 세션 1: 구글 키워드 광고로 시작한 세션
  - 세션 2: 페이스북 광고를 통해 생긴 세션
  - 세션 3: 네이버 광고를 통해 생긴 세션
![rdb-example-user-100](/assets/images/rdb-example-user-100.png){: .align-center .img-80}

## 관계형 데이터베이스 예제 - 데이터베이스와 테이블
![rdb-example-user-table](/assets/images/rdb-example-user-table.png){: .align-center .img-80}
![user-session-channel](/assets/images/user-session-channel.png){: .align-center .img-40}
![session-timestamp](/assets/images/session-timestamp.png){: .align-center .img-40}

## SQL 소개
- SQL: Structured Query Language
- SQL은 1970년대 초반에 IBM 이 개발한 구조화된 데이터 질의 언어
  - 주로 관계형 데이터베이스에 있는 데이터를 질의 하는 언어
- 두 종류의 언어로 구성됨
  - DDL (Data Definition Language)
    - 테이블의 구조를 정의하는 언어
  - DML (Data Manipulation Language)
    - 테이블에 레코드를 추가/삭제/갱신 해주는데 사용하는 언어

### SQL DDL - 테이블 구조 정의 언어
- `CREATE TABLE`
- `DROP TABLE`
- `ALTER TABLE`
- `raw_data` 폴더에 `user_session_channel` 라는 이름인 테이블을 생성
  ```sql
  CREATE TABLE raw_data.user_session_channel (
    userid int,
    sessionid varchar(32),
    channel varchar(32)
  );
  ```

### SQL DML - 테이블 데이터 조작 언어
- `SELECT FROM`: 테이블에서 레코드와 필드를 읽어오는데 사용
  - 보통 다수의 테이블의 조인해서 사용
- `INSERT FROM`: 테이블에 레코드를 추가하는데 사용
- `UPDATE FROM`: 테이블의 레코드를 수정
- `DELETE FROM`: 테이블에서 레코드를 삭제
  ```sql
  SELECT 필드이름1, 필드이름2, ...
  FROM 테이블이름
  WHERE 선택조건
  ORDER BY 필드이름 [ASC|DESC]
  LIMIT N
  ```

  ```sql
  SELECT * FROM raw_data.user_session_channel LIMIT 10;
  ```
  ```sql
  SELECT COUNT(1) FROM raw_data.user_session_channel;
  ```
  ```sql
  SELECT COUNT(1) FROM raw_data.user_session_channel
  WHERE channel = 'Facebook'; --channel 이름이 Facebook인 경우만
  ```
  ```sql
  SELECT COUNT(1) FROM raw_data.user_session_channel
  WHERE channel ilike '%o%'; --channel 이름에 o나 O가 있는 경우만 (%: filesystem 에서 *에 해당)
  ```
  ```sql
  SELECT channel, COUNT(1) --channel 별로 레코드수 카운트하기
  FROM raw_data.user_session_channel
  GROUP BY channel;
  ```
- 세션에 대한 모든 정보 읽어오기: user_session_channel 과 session_timestamp 조인하기
```sql
SELECT *
FROM raw_data.user_session_channel usc
JOIN raw_data.session_timestamp ts ON usc.sessionID = ts.sessionID;
```

# SQL 실습 - Redshift 기반 SQL 실습
# SparkSQL 이란
# SparkSQL 실습







# Appendix
## Reference
> Hadoop Name node, Data node <https://www.geeksforgeeks.org/hadoop-hdfs-hadoop-distributed-file-system/>  
