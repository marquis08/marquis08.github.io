---
date: 2021-07-07 02:13
title: "SQL Analysis - GROUP BY & CTAS"
categories: DevCourse2 SQL MathJax DevCourse2_SQL
tags: DevCourse2 SQL MathJax DevCourse2_SQL
## 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# GROUP BY와 AGGREGATE 함수
- 테이블의 레코드를 그룹핑하여 그룹별로 다양한 정보를 계산
- 이는 두 단계로 이루어짐
    - 그룹핑을 할 필드를 결정 (하나 이상의 필드가 될 수 있음)
        - `GROUP BY`로 지정 (필드 이름을 사용하거나 필드 일련번호를 사용)
    - 다음 그룹별로 계산할 내용을 결정
        - 여기서 **Aggregate** 함수를 사용
        - `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `LISTAGG`, ...
            - 보통 필드 이름을 지정하는 것이 일반적 (alias)

- 월별 세션수를 계산하는 SQL
    - `raw_data.session_timestamp` 를 사용 (sessionId 와 ts 필드)
- 
```sql
SELECT
    LEFT(ts, 7) AS mon,
    COUNT(1) AS session_count
FROM raw_data.session_timestamp
GROUP BY 1 -- GROUP BY mom, GROUP BY LEFT(ts, y)
ORDER BY 1;
```


## 문제 풀이
- `raw_data.session_timestamp`와 `raw_data.user_session_channel` 테이블들을 사용
- 다음을 계산하는 SQL을 만들어보자
    - 가장 많이 사용된 채널은 무엇인가
    - 가장 많은 세션을 만들어낸 사용자 ID는 무엇인가
    - 월별 유니크한 사용자 수 (MAU - Monthly Active User)
        - 한 사용자는 한번만 카운트
    - 월별 채널별 유니크한 사용자 수

### 가장 많이 사용된 채널
- 가장 많이 사용되었다는 정의
    - 사용자 기반 아니면 세션 기반
- 필요한 정보
    - 채널 정보, 사용자 정보 또는 세션 정보
- 먼저 어느 테이블을 사용해야 하는지 생각
    - user_session_channel
    - session_timestamp
    - 혹은 이 2개의 테이블을 조인해야 하나
- 
```sql
SELECT
    channel,
    COUNT(1) AS session_count,
    COUNT(DISTINCT userId) AS user_count
FROM raw_data.user_session_channel
GROUP BY 1 -- `GROUP BY channel`
ORDER BY 2 DESC; -- `ORDER BY session_count DESC`
```  



### 가장 많은 세션을 만들어낸 사용자 ID는 무엇인가
- 필요한 정보
    - 사용자 정보, 세션 정보
- 먼저 어느 테이블을 사용해야 하는지 생각
    - user_session_channel
    - session_timestamp
    - 혹은 이 2개의 테이블을 조인해야 하나
- 
```sql
SELECT
    userId,
    COUNT(1) AS count
FROM raw_data.user_session_channel
GROUP BY 1 -- `GROUP BY userId`
ORDER BY 2 DESC -- `ORDER BY count DESC`  
LIMIT 1;
```  

### 월별 유니크한 사용자 수
MAU - Monthly Active User 에 해당
- 필요한 정보
    - 시간 정보, 사용자 정보
- 먼저 어느 테이블을 사용해야 하는지 생각
    - user_session_channel(userId, sessionId, channel)
    - session_timestamp(sessionId, ts)
    - 혹은 이 2개의 테이블을 조인해야 하나
- 
```sql
SELECT
    TO_CHAR(A.ts 'YYYY-MM') AS month,
    COUNT(DISTINCT B.userid) as mau
FROM raw_data.session_timestamp A
JOIN raw_data.user_session_channel B ON A.sessionid = B.sessionid -- `inner join`
GROUP BY 1 -- `GROUP BY 1` == `GROUP BY month` == `GROUP BY TO_CHAR(A.ts, 'YYYY-MM')`
ORDER BY 1 DESC;
```  

- 
```sql
TO_CHAR(A.ts 'YYYY-MM')
```  
- `TO_CHAR` 와 같은 기능
    - `LEFT(A.ts, 7)`
    - `DATE_TRUNC('month', A.ts)`
    - `SUBSTRING(A.ts, 1, 7)`

- `COUNT`의 동작을 잘 이해하는 것이 중요
    - `DISTINCT`가 없으면 모든 레코드
    - `DISTINCT`한 userid만
- `JOIN` 앞에 아무것도 없으면 `INNER JOIN`

- `GROUP BY`와 `ORDER BY`
    - 포지션 번호 혹은 필드이름
- `GROUP BY 1` == `GROUP BY month` == `GROUP BY TO_CHAR(A.ts, 'YYYY-MM')`

### 월별 채널별 유니크한 사용자 수
- 필요한 정보
    - 시간 정보, 사용자 정보, 채널정보
- 먼저 어느 테이블을 사용해야 하는지 생각
    - user_session_channel(userId, sessionId, channel)
    - session_timestamp(sessionId, ts)
    - 혹은 이 2개의 테이블을 조인해야 하나
- 
```sql
SELECT
    TO_CHAR(A.ts 'YYYY-MM') AS month,
    channel,
    COUNT(DISTINCT B.userid) as mau
FROM raw_data.session_timestamp A
JOIN raw_data.user_session_channel B ON A.sessionid = B.sessionid
GROUP BY 1, 2
ORDER BY 1 DESC, 2; -- 월별 order, 채널별 order
```  

# CTAS와 CTE 소개
## CTAS (CREATE TABLE AS SELECT)
SELECT를 가지고 테이블 생성
- 간단하게 새로운 테이블을 만드는 방법
- 자주 조인하는 테이블들이 있다면 이를 CTAS를 사용해서 조인해두면 편리해짐

- CTAS로 `adhoc.keeyong_session_summary` 테이블 생성
```sql
DROP TABLE IF EXISTS adhoc.keeyong_session_summary;
CREATE TABLE adhoc.keeyong_session_summary AS
SELECT B.*, A.ts FROM raw_data.session_timestamp A
JOIN raw_data.user_session_channel B ON A.sessionid = B.sessionid;
```  

## 월별 유니크한 사용자 수를 다시 풀어보기
```sql
SELECT
    TO_CHAR(ts, 'YYYY-MM') AS mouth,
    COUNT(DISTINCT userid) AS mau
FROM adhoc.keeyong_session_summary
GROUP BY 1
ORDER BY 1 DESC;
```  
> - 
```sql
SELECT
    TO_CHAR(A.ts 'YYYY-MM') AS month,
    COUNT(DISTINCT B.userid) as mau
FROM raw_data.session_timestamp A
JOIN raw_data.user_session_channel B ON A.sessionid = B.sessionid
GROUP BY 1
ORDER BY 1 DESC;
```  
>  
> 기존에 했던 `JOIN` 부분을 안써도 됨.  
> 이렇게 하는 장점은 복잡한 경우에 헷갈림을 방지할 수 있음.  


## 항상 시도해봐야하는 데이터 품질 확인 방법들
- 중복된 레코드들 체크하기
- 최근 데이터의 존재여부 체크하기(freshness)
- PK uniqueness 가 지켜지는지 체크
- 값이 비어있는 컬럼들이 있는지 체크

### 중복된 레코드들 체크하기
#### 다음 두 개의 카운트를 비교
```sql
SELECT COUNT(1)
FROM adhoc.keeyong_session_summary;
```

```sql
SELECT COUNT(1)
FROM (
    SELECT DISTINCT userId, sessionId, ts, channel
    FROM adhoc.keeyong_session_summary;
);
```  

위 처럼 `FROM` 에서 항상 physical한 table이 오지 않아도 상관없음.  
위 에서는 `FROM` 안에 select를 nest 한 형태  


#### CTE 사용해서 중복 제거 후 카운트 (CTE: Common Table Expression)
> CTE: 재사용 가능한 임시테이블 만들기  

```sql
With ds AS (
    SELECT DISTINCT userId, sessionId, ts, channel
    FROM adhoc.keeyong_session_summary
)
SELECT COUNT(1)
FROM ds;
```  

```sql
SELECT COUNT(1)
FROM (
    SELECT DISTINCT userId, sessionId, ts, channel
    FROM adhoc.keeyong_session_summary;
);
```  
- CTE를 사용한 코드와 From에 nest를 사용한 코드는 똑같지만 CTE를 사용해서 위로 빼놓는 것이 좋다.
    - CTE를 사용해서 임시테이블을 만들어 놓은 것을 뒤에서 반복해서 재사용할 경우 유용하기 때문  


### 최근 데이터의 존재 여부 체크하기 (freshness)
```sql
SELECT MIN(ts), MAX(ts)
FROM adhoc.keeyong_session_summary;
```  

### PK uniqueness 지켜지는지 체크
```sql
SELECT sessionId, COUNT(1)
FROM adhoc.keeyong_session_summary
GROUP BY 1
ORDER BY 2 DESC
LIMIT 1;
```  
- `sessionId` 로 GROUP BY 하면, 같은 `sessionId`를 갖는 레코드들이 모이고, 그것들을 `COUNT(1)` 해서, `COUNT(1)` 를 기준으로 내림차순을 하면, return 된 레코드의 COUNT 수가 1보다 크면 중복이 있다는 의미니까.  


### 값이 비어있는 컬럼들이 있는지 체크하기
```sql
SELECT
    COUNT(CASE WHEN sessionId is NULL THEN 1 END) sessionid_null_count,
    COUNT(CASE WHEN userId is NULL THEN 1 END) userid_null_count,
    COUNT(CASE WHEN ts is NULL THEN 1 END) ts_null_count,
    COUNT(CASE WHEN channel is NULL THEN 1 END) channel_null_count,
FROM adhoc.keeyong_session_summary;
```  
- `NULL` 일 경우만 `COUNT`

# 숙제
지금까지 session_timestamp와 user_session_channel을 사용

## 채널별 월 매출액 테이블 만들기 (본인 스키마 밑에 CTAS로 테이블 만들기)
- session_timestamp, user_session_channel, session_transaction 테이블들 사용
- 아래와 같은 필드로 구성
    - month
    - channel
    - uniqueUsers(총방문 사용자)
    - paidUsers(구매 사용자: refund한 경우도 판매로 고려)
    - conversionRate(구매 사용자/ 총방문 사용자)
    - grossRevenue(refund 포함)
    - netRevenue(refund 제외)
