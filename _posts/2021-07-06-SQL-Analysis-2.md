---
date: 2021-07-06 03:45
title: "SQL Analysis - SELECT"
categories: DevCourse2 SQL MathJax
tags: DevCourse2 SQL MathJax
## 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Redshift 론치 데모
# 예제 테이블 소개
- 사용자 ID: 웹서비스에서 등록된 사용자마다 부여하는 유일한 ID
- 세션 ID: 세션마다 부여되는 ID
    - 세션: 사용자의 방문을 논리적인 단위로 나눈 것
        - 사용자가 방문 후 30분간 interaction이 없으면 종료, 다시 사용하는 경우 새로운 세션 생성(time bound)
        - 사용자가 외부 링크(광고)를 타고 오거나 직접 방문해서 올 경우 세션을 생성
    - 하나의 사용자는 여러 개의 세션을 가질 수 있음
    - 보통 세션의 경우 세션을 만들어낸 접점을 채널이란 이름으로 기록해둠.
        - 마케팅 관련 기여도 분석을 위함
    - 세션이 생긴 시간도 기록함.
- 이 정보를 기반으로 다양한 데이터 분석과 지표 설정이 가능
    - 마케팅 관련, 사용자 트래픽 관련
    - DAU, WAU, MAU 등의 일/주/월별 Activae User 차트
    - Marketing Channel Attribution 분석
        - 어느 채널에 광고를 하는 것이 가장 효과적인가

# SQL 소개 (DDL, DML)
## SQL 기본
- 다수의 SQL 문을 실행한다면 `;` 으로 분리 필요
    - `SQL 문1;`,`SQL 문2;`,`SQL 문3;`
- SQL 주석
    - `--` : 인라인 한줄짜리 주석
    - `/*--*/` : 여러 줄에 걸쳐 사용 가능
- SQL 키워드는 대문자를 사용한다던지 하는 나름대로의 포맷팅이 필요
    - 팀프로젝트라면 팀에서 사용하는 공통 포맷이 필요
- 테이블/필드이름의 명명규칙을 정하는 것이 중요
    - 단수형 vs. 복수형
        - User vs. Users
    - _ vs. CamelCasing
        - user_session_channel vs. UserSessionChannel  

## SQL DDL : 테이블 구조 정의 언어
### CREATE TABLE
- Primary key 속성을 지정할 수 있으나 무시됨
    - Primary key uniqueness
        - Big DW 에서는 지켜지지 않음 (Redshift, BigQuery, Snowflake)
- CTAS: `CREATE TABLE table_name AS SELECT`
    - `SELECT` 를 통해서 읽어들인 레코드들을 통해 table_name이라는 테이블을 생성
    - vs. `CREATE TABLE and then INSERT`
- 
```sql
CREATE TABLE raw_data.user_session_channel(
    userid int,
    sessionid varchar(32) primary key,
    channel varchar(32)
)
```  

### DROP TABLE
- DROP TABLE table_name;
    - 없는 테이블을 지우려고 하는 경우 에러발생
- `DROP TABLE IF EXIST table_name;`
- vs. `DELETE FROM`
    - `DELETE FROM`은 조건에 맞는 레코들을 지움 (테이블 자체는 존재)  

### ALTER TABLE
- 새로운 컬럼 추가:
    - `ALTER TABLE 테이블이름 ADD COLUMN 필드이름 필드타입;`
- 기존 컬럼 이름 변경:
    - `ALTER TABLE 테이블이름 RENAME 현재필드이름 to 새필드이름`
- 기존 컬럼 제거:
    - `ALTER TABLE 테이블이름 DROP COLUMN 필드이름;`
- 테이블 이름 변경:
    - `ALTER TABLE 현재테이블이름 RENAME to 새테이블이름;`

## SQL DML : 테이블 데이터 조작 언어
### 레코드 질의 언어: SELECT
- `SELECT FROM`: 테이블에서 레코드와 필드를 읽어오는데 사용
- `WHERE를` 사용해서 레코드 선택조건을 지정
- `GROUP BY`를 통해 정보를 그룹 레벨에서 뽑는데 사용하기도 함
    - DAU, WAU, MAU 계산은 `GROUP BY` 필요
- `ORDER BY`를 사용해서 레코드 순서 결정
- 다수의 테이블을 `JOIN`해서 사용하기도  

### 레코드 수정 언어
- `INSERT INTO`: 테이블에 레코드를 추가하는데 사용
- `UPDATE FROM`: 테이블 레코드의 필드 값 수정
- `DELETE FROM`: 테이블에서 레코드를 삭제
    - vs. `TRUNCATE`: `transaction` 사용불가(DELETE FROM은 가능)  

# SQL 실습 환경 소개
- 현업에서 깨끗한 데이터란 존재하지 않음
    - 항상 데이터를 믿을 수 있는지 의심할 것
    - 실제 레코드를 몇 개 살펴보는 것 만한 것이 없음 &rArr; 노가다
- 데이터 일을 한다면 항상 데이터의 품질을 의심하고 체크하는 버릇이 필요
    - 중복된 레코드들 체크하기
    - 최근 데이터의 존재 여부 체크하기 (freshness)
    - Primary Key uniqueness가 지켜지는지 체크하기
    - 값이 비어있는 컬럼들이 있는지 체크하기
    - 위의 체크는 코딩의 unit test 형태도 만들어 매번 쉽게 체크해볼 수 있음
- 어느 시점이 되면 너무나 많은 테이블들이 존재
    - 회사 성장과 밀접한 관련
    - 중요 테이블들이 무엇이고 그것들의 메타 정보를 잘 관리하는 것이 중요
- 그 시점부터는 DATA Discovery 문제들이 생김
    - 무슨 테이블에 내가 원하고 신뢰할 수 있는 정보가 들어있나
    - 테이블에 대해 질문을 하고 싶은데 누구에게 질문을 해야하나
- 이 문제를 해결하기 위한 다양한 오픈소스와 서비스(SaaS)들이 출현
    - DataHub(LinkedIn), Amundsen(Lyft), ...
    - Select Start, DataFrame, ...
    - 각 테이블 별로 사용횟수와 최근 사용자, 최다 사용자가 나와서 누가 주로 사용하는지 알 수 있음.


# SELECT 소개
- 테이블에서 레코드들을 읽어오는데 사용
- WHERE를 사용해 조건을 만족하는 레코드 읽음
- 
```sql
SELECT 필드이름1, 필드이름2
FROM 테이블이름
WHERE 선택조건
GROUP BY 필드이름1, 필드이름2
ORDER BY 필드이름 ASC 필드이름 DESC
LIMIT N;
```  
- `ORDER BY`: 필드이름 대신에 숫자 사용 가능
- 
```sql
SELECT *
FROM raw_data.user_session_channel;
```  

- 유일한 채널 이름을 알고 싶은 경우
    - 
    ```sql
    SELECT DISTINCT channel
    FROM raw_data.user_session_channel;
    ```  
- 채널별 카운트를 하고 싶은 경우. `COUNT`
    - 
    ```sql
    SELECT channel, COUNT(1)
    FROM raw_data.user_session_channel
    GROUP BY 1;
    ```  
    - `GROUP BY 1;` ordinal 하게 표현 여기서는 channel이 첫번째 필드이므로
- 테이블의 모든 레코드 수 카운트. COUNT(*). 하나의 레코드
    - 
    ```sql
    SELECT COUNT(1)
    FROM raw_data.user_session_channel;
    ```
    - `GROUP BY` 없이 사용하면, SELECT의 조건을 만족하는 레코드를 불러옴.
- channel 이름이 Facebook 인 겨우만 고려해서 레코드 수 카운트
    - 
    ```sql
    SELECT COUNT(1)
    FROM raw_data.user_session_channel
    WHERE channel = 'Facebook';
    ```  

## CASE WHEN
- 필드 값의 변환을 위해 사용 가능
    - CASE WHEN 조건 THEN 참일 때 값 ELSE 거짓일 때 값 END 필드이름
- 여러 조건을 사용하여 변환 가능
- 
```sql
CASE
    WHEN 조건1 THEN 값1
    WHEN 조건2 THEN 값2
    ELSE 값3
END 필드이름
```  
- 
```sql
SELECT CASE
    WHEN channel in ('Facebook', 'Instagram') THEN 'Social-Media'
    WHEN channel in ('Google','Naver') THEN 'Search-Engine'
    ELSE 'Something-Else'
END channel_type
FROM raw_data.user_session_channel;
```  

## NULL
- 필드 지정시 값이 없는 경우 NULL로 지정 가능
    - 테이블 정의시 디폴트 값으로도 지정 가능
- 어떤 필드의 값이 NULL인지 아닌지 비교
    - `field1 is NULL`
    - `field1 is not NULL`
- NULL 이 사칙연산에 사용되면 &rArr; NULL
    - `SELECT 0+NULL`

## COUNT 함수 제대로 이해하기
|value|
|------|
|NULL|
|1|
|1|
|0|
|0|
|4|
|3|

테이블: count_test

- SELECT COUNT(1) FROM count_test &rArr; 7
    - count all
- SELECT COUNT(0) FROM count_test &rArr; 7
    - count all
- SELECT COUNT(NULL) FROM count_test &rArr; 0
    - NULL인 경우는 안세고 not NULL 인 경우 셈.
- SELECT COUNT(value) FROM count_test &rArr; 6
    - count not NULL
- SELECT COUNT(DISTINCT value) FROM count_test &rArr; 4
    - DISTINCT 했기 때문에 5개(NULL, 1, 0, 4, 3) 중에서 not null인 것들을 count하면 4
    - Count Distinct values except NULL  

## WHERE
### IN
- `WHERE channel in ('Google','Youtube')`
    - WHERE channel = 'Google' OR channel = 'Youtube'
- `NOT IN`
    - `WHERE channel not in ('Google','Youtube')`  

### LIKE and ILIKE
- `LIKE` is a case sensitive string match.
- `ILKIKE` is a case-insensitive string match.
- `WHERE channel LIKE 'G%'` &rarr; `'G*'`
    - G로 시작하는 모든 것
- `WHERE channel LIKE '%o%'` &rarr; `'*o*'`
    -  o가 들어간 모든 것.
- `NOT LIKE` or `NOT LIKE`  

### BETWEEN
- Used for date range matching
- timestamp  

> 위의 오퍼레이터 들은 CASE WHEN 사이에서도 사용가능  

### 예제
- WHERE 조건을 만족하는 레코드 카운팅
- - 
```sql
SELECT COUNT(1)
FROM raw_data.user_session_channel
WHERE channel in ('Google','Facebook');
```

- - 
```sql
SELECT COUNT(1)
FROM raw_data.user_session_channel
WHERE channel ilike 'Google' or channel ilike 'Facebook';
```

- WHERE 조건을 만족하는 channel이 DISTINCT한 레코드가 무엇이 있는지
- - 
```sql
SELECT DISTINCT channel
FROM raw_data.user_session_channel
WHERE channel ILIKE '%o%';
```

- - 
```sql
SELECT DISTINCT channel
FROM raw_data.user_session_channel
WHERE channel NOT ILIKE '%o%';
```

## STRING Functions
- LEFT(str, N)
- REPLACE(str, exp1, exp2)
- UPPER(str)
- LOWER(str)
- LEN(str)
- LPAD, RPAD
- SUBSTRING
- 
```sql
SELECT
    LEN(channel),
    UPPER(channel),
    LOWER(channel),
    LEFT(channel, 4)
FROM raw_data.user_session_channel;
```  
- `LEFT(channel, 4)`: channel 필드에서 문자열 처음 4개만
- `REPLACE(str, exp1, exp2)`: `str`에서 `exp1`을 찾아서 `exp2`로 바꿈
- `LPAD, RPAD`: string을 padding
- `SUBSTRING`: 지정한 시작점을 기준으로 N개만큼  

## ORDER BY
- Default ordering is ascending
    - `ORDER BY 1 ASC`
- Descending requires "DESC"
    - `ORDER BY 1 DESC`
- Ordering by multiple columns
    - `ORDER BY 1 DESC, 2, 3`
- NULL 값 순서는
    - NULL 값들은 오름차순일 경우 (ASC), 마지막에 위치함
    - NULL 값들은 내림차순 일 경우 (DESC) 처음에 위치함
    - 이를 바꾸고 싶다면 NULLS FIRST 혹은 NULLS LAST를 사용  

## 타입 변환
- DATE Conversion
    - Time Zone 변환
        - list up timezone
            - `select pg_timezone_names();`
        - CONVERT_TIMEZONE('America/Los_Angeles',ts)
            - ts: utc timestamp
    - DATE, TRUNCATE
        - ts를 인자로 받아서 년도나 날짜로 return. (시분초는 삭제)
    - DATE_TRUNC
        - 첫번째 인자가 어떤 값을 추출하는지 지정(week, month, day, ...)
    - EXTRACT or DATE_PART
        - 날짜시간에서 특정 부분의 값을 추출
    - DATEDIFF
    - DATEADD
    - GET_CURRENT, ...
- TO_CHAR, TO_TIMESTAMP  

## Type Casting
- 1/2의 결과는
    - 0이 됨. 정수간의 연산은 정수
        - 분자나 분모 중의 하나를 float으로 casting해야 0.5가 나옴
        - 이는 프로그래밍 언어에서도 일반적으로 동일하게 동작
- 오퍼레이터 사용
    - `file_name::type`
    - ex. `category::float` &rArr; category필드를 float타입으로
- cast 함수 사용
    - `cast(category as float)`
