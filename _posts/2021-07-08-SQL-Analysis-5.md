---
date: 2021-07-08 00:09
title: "SQL Analysis - 트랜잭션과 기타 고급 SQL 문법"
categories: DevCourse2 SQL MathJax
tags: DevCourse2 SQL MathJax
## 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# 숙제1 리뷰
## 사용자별로 처음 채널과 마지막 채널 알아내기
- ROW_NUMBER vs. FIRST_VALUE/LAST_VALUE
- 사용자 251번의 시간순으로 봤을 때 첫 번째 채널과 마지막 채널은 무엇인가?
    - 아래 쿼리를 실행해서 처음과 마지막 채널을 보면 됨
    - 
    ```sql
    SELECT ts, channel
    FROM raw_data.user_session_channel usc
    JOIN raw_data.session_timestamp  st ON usc.sessionid = st.sessionid
    WHERE userid = 251
    ORDER BY 1
    ```
- `ROW_NUMBER` 를 이용해서 해보자
    - 일종의 GROUPING 방법
    - 일련번호를 붙이는 방법
    - ROW_NUMBER() OVER (PARTITION BY field1 ORDER BY field2) nn  
    - `PARTITION BY`: 어떤 필드를 기준으로 GROUPING 할건지
    - `ORDER BY`: 일련번호 붙이는 기준
    

## CTE 를 빌딩블록으로 사용
```sql
WITH first AS (
    SELECT userid, ts, channel, ROW_NUMBER() OVER(PARTITION BY userid ORDER BY ts) seq
    FROM raw_data.user_session_channel usc
    JOIN raw_data.session_timestamp st ON usc.sessionid = st.sessionid
), last AS (
    SELECT userid, ts, channel, ROW_NUMBER() OVER(PARTITION BY userid ORDER BY ts DESC) seq
    FROM raw_data.user_session_channel usc
    JOIN raw_data.session_timestamp st ON usc.sessionid = st.sessionid
)
SELECT first.userid AS userid, first.channel AS first_channel, last.channel AS last_channel
FROM first
JOIN last ON first.userid = last.userid and last.seq = 1
WHERE first.seq = 1;
-- 949 rows affected.
-- userid 	first_channel 	last_channel
-- 27 	Youtube 	Instagram
-- 29 	Naver 	Naver
-- 33 	Google 	Youtube
-- 40 	Youtube 	Google
-- 44 	Naver 	Instagram
-- 59 	Instagram 	Instagram
-- ...
-- 232 	Instagram 	Naver
-- 243 	Youtube 	Instagram
-- 248 	Naver 	Instagram
-- 251 	Facebook 	Google
-- 253 	Facebook 	Youtube
-- 255 	Facebook 	Instagram
-- 258 	Google 	Facebook
```  
- 
```sql
first AS (
    SELECT userid, ts, channel, ROW_NUMBER() OVER(PARTITION BY userid ORDER BY ts) seq
    FROM raw_data.user_session_channel usc
    JOIN raw_data.session_timestamp st ON usc.sessionid = st.sessionid
)
```
- - 모든 사용자별로 첫번쨰 channel을 return.
- - `user_session_channel` **JOIN** `session_timestamp`
- - ROW_NUMBER() OVER(PARTITION BY userid ORDER BY ts) seq: 일련번호 붙이기(오래전에 생성된 순)
- - `last` table은 `ORDER BY ts DESC` 이것만 다른 것.

- `CTE`로 만든 `first`, `last` 테이블을 다시 JOIN 함.
    - JOIN 시, userid 가 같고(`first.userid = last.userid`), `last.seq = 1`인 경우(마지막 레코드)만.
    - WHERE 문에서 `first.seq = 1`으로 줘서 첫번째 레코드만
    - 
    ```sql
    JOIN last ON first.userid = last.userid and last.seq = 1 first.seq = 1;
    ```
    - 이런식으로도 가능하지만, 일반적으로 `FROM` 테이블에서 사용하는 필드는 `WHERE` 에서 붙임 ✅

## JOIN 방식
```sql
SELECT first.userid AS userid, first.channel AS first_channel, last.channel AS last_channel
FROM (
    SELECT userid, ts, channel, ROW_NUMBER() OVER(PARTITION BY userid ORDER BY ts) seq
    FROM raw_data.user_session_channel usc
    JOIN raw_data.session_timestamp st ON usc.sessionid = st.sessionid
) first
JOIN (
    SELECT userid, ts, channel, ROW_NUMBER() OVER(PARTITION BY userid ORDER BY ts DESC) seq
    FROM raw_data.user_session_channel usc
    JOIN raw_data.session_timestamp st ON usc.sessionid = st.sessionid
)       last ON first.userid = last.userid and last.seq = 1
WHERE first.seq = 1;
```  
- `CTE` 대신 `first`, `last`를 `FROM` 과 `JOIN` 으로 치환한 것.


## GROUP BY 방식
```sql
SELECT userid,
    MAX(CASE WHEN rn1 = 1 THEN channel END) first_touch,
    MAX(CASE WHEN rn2 = 1 THEN channel END) last_touch
FROM (
    SELECT userid,
        channel,
        (ROW_NUMBER() OVER (PARTITION BY usc.userid ORDER BY  st.ts asc)) AS rn1,
        (ROW_NUMBER() OVER (PARTITION BY usc.userid ORDER BY  st.ts desc)) AS rn2
    FROM raw_data.user_session_channel usc
    JOIN raw_data.session_timestamp st ON usc.sessionid = st.sessionid
)
GROUP BY 1;
```  

## FIRST_VALUE/LAST_VALUE (제일 간단한 방법)
```sql
SELECT DISTINCT
    A.userid,
    FIRST_VALUE(A.channel) over(partition by A.userid order by B.ts
rows between unbounded preceding and unbounded following) AS First_Channel,
    LAST_VALUE(A.channel) over(partition by A.userid order by B.ts
rows between unbounded preceding and unbounded following) AS Last_Channel
FROM raw_data.user_session_channel A
LEFT JOIN raw_data.session_timestamp B ON A.sessionid = B.sessionid
ORDER BY 1;
```  
- `FIRST_VALUE`, `LAST_VALUE`

# 숙제2 리뷰
## Gross Revenue 가 가장 큰 UserID 10 개 찾기
- `user_session_channel` 과 `session_transaction` 과 `session_timestamp` 테이블을 사용
- `Gross revenue`: `Refund` 포함한 매출  
    - `session_transaction` 에 `amount` 필드가 있음
    - `Gross revenue`는 refund 여부 상관없이 합친 것.
    - `Net revenue`는 refund 여부 빼고 합친 것.

## GROUP BY
```sql
SELECT
    userID,
    SUM(amount)
FROM raw_data.session_transaction st
LEFT JOIN raw_data.user_session_channel usc ON st.sessionid = usc.sessionid
GROUP BY 1
ORDER BY 2 DESC
LIMIT 10;
-- 10 rows affected.
-- userid 	sum
-- 989 	743
-- 772 	556
-- 1615 	506
-- 654 	488
-- 1651 	463
-- 973 	438
-- 262 	422
-- 1099 	421
-- 2682 	414
-- 891 	412
```  
- `LIMIT 10`: 동점자 중에 하나만 나옴
    - 동점자들도 나오게 하려면?

## SUM OVER (Window Function)
```sql
SELECT DISTINCT
    usc.userid,
    SUM(amount) OVER(PARTITION BY usc.userid)
FROM raw_data.user_session_channel AS usc
JOIN raw_data.session_transaction AS revenue ON revenue.sessionid = usc.sessionid  
ORDER BY 2 DESC 
LIMIT 10;
-- 10 rows affected.
-- userid 	sum
-- 989 	743
-- 772 	556
-- 1615 	506
-- 654 	488
-- 1651 	463
-- 973 	438
-- 262 	422
-- 1099 	421
-- 2682 	414
-- 891 	412
```
- 같은 `userid` 를 같은 레코드들을 묶어서 그것들의 `amount` 를 `sum`
- 위의 `GROUP BY` 를 사용하면 `userid` 당 하나만 들어가지만, SUM() OVER 를 사용하게 되면 동일 userid 가 다 나오게 됨.
    - `DISTINCT` 를 사용한 이유.
    - `DISTINCT` 를 안쓸 경우
    ```sql
    -- 10 rows affected.
    -- userid 	sum
    -- 989 	743
    -- 989 	743
    -- 989 	743
    -- 989 	743
    -- 989 	743
    -- 989 	743
    -- 989 	743
    -- 989 	743
    -- 772 	556
    -- 772 	556
    ```

# 숙제3 리뷰
## raw_data.nps 테이블을 바탕으로 월별 NPS 계산
- 고객들이 0 (의향없음) 에서 10 (의향 아주 높음)
- `detractor` (비추천자): 0 에서 6
- `passive` (소극자): 7이나 8점
- `promoter` (홍보자): 9 나 10점
- `NPS` = promoter 퍼센트 - detractor 퍼센트  
- 1번쨰 Solution
    - 
    ```sql
    SELECT month, 
    ROUND((promoters-detractors)::float/total_count*100, 2) AS overall_nps --float으로 캐스팅 안하면 integer 0가 되버림. NULLIF 함수 사용한거 생각.
    FROM (
    SELECT LEFT(created, 7) AS month,
        COUNT(CASE WHEN score >= 9 THEN 1 END) AS promoters,
        COUNT(CASE WHEN score <= 6 THEN 1 END) AS detractors,
        COUNT(CASE WHEN score > 6 AND score < 9 THEN 1 END) As passives,
        COUNT(1) AS total_count
    FROM raw_data.nps
    GROUP BY 1
    ORDER BY 1
    );
    -- month 	overall_nps
    -- 2019-01 	2.36
    -- 2019-02 	30.54
    -- 2019-03 	52.91
    -- 2019-04 	53.0
    -- 2019-05 	54.52
    -- 2019-06 	65.02
    -- 2019-07 	64.51
    -- 2019-08 	67.71
    -- 2019-09 	37.95
    -- 2019-10 	53.29
    -- 2019-11 	61.29
    -- 2019-12 	65.99
    ```
- 2번째 Solution
    - 
    ```sql
    SELECT LEFT(created, 7) AS month,
    ROUND(SUM(CASE
        WHEN score >= 9 THEN 1 
        WHEN score <= 6 THEN -1 END)::float*100/COUNT(1), 2)
    FROM raw_data.nps
    GROUP BY 1
    ORDER BY 1;
    -- month 	round
    -- 2019-01 	2.36
    -- 2019-02 	30.54
    -- 2019-03 	52.91
    -- 2019-04 	53.0
    -- 2019-05 	54.52
    -- 2019-06 	65.02
    -- 2019-07 	64.51
    -- 2019-08 	67.71
    -- 2019-09 	37.95
    -- 2019-10 	53.29
    -- 2019-11 	61.29
    -- 2019-12 	65.99
    ```


# 트랜잭션 소개와 실습
## 트랜잭션이란
- `Atomic` 하게 실행되어야 하는 SQL들을 묶어서 하나의 작업처럼 처리하는 방법
    - 이는 `DDL` 이나 `DML` 중 레코드를 수정/추가/삭제한 것에만 의미가 있음
    - `SELECT` 에는 트랜잭션을 사용할 이유가 없음
    - `BEGIN` 과 `END` 혹은 `BEGIN` 과 `COMMIT` 사이에 해당 SQL 들을 사용
    - `ROLLBACK`
        - `BEGIN` 과 `END` 사이에 sql이 하나라도 실패하면 원래 상태로 돌아감.
- 은행 계좌 이체가 아주 좋은 예
    - 계좌 이체: 인출과 입금의 두 과정으로 이뤄짐
    - 만일 인출은 성공했는데 입금이 실패한다면?
    - 이 두 과정은 동시에 성공하던지 실패해야함 &rArr; Atomic 하다는 의미
    - 이런 과정들을 트랜잭션으로 묶어주어야함
    - 조회만 한다면 이는 트랜잭션으로 묶일 이유가 없음
```sql
BEGIN;
    A의 계좌로부터 인출;
    B의 계좌로 입금;
END;
```
- `END` 와 `COMMIT` 은 동일
- 만일 `BEGIN` 전의 상태로 돌아가고 싶다면 `ROLLBACK` 실행

- 이 동작은 `commit mode` 에 따라 달라짐  

## 트랜잭션 커밋 모드: autocommit
- `autocommit = True`
    - 모든 레코드 수정/삭제/추가 작업이 기본적으로 바로 `DB` 에 쓰여짐. 이를 `commit` 된다고 함.
    - 만일 특정 작업을 트랜잭션으로 묶고 싶다면 BEING 과 END(COMMIT)/ROLLBACK 으로 처리
- `autocommit = False`
    - 모든 레코드 수정/삭제/추가 작업이 COMMIT 호출될 때까지 커밋되지 않음  

## 트랜잭션 방식
- `Google Colab` 의 트랜잭션
    - 기본적으로 모든 `SQL statement` 가 바로 커밋됨 (`autocommit = True`)
    - 이를 바꾸고 싶다면 `BEGIN;END;` 혹은 `BEGIN;COMMIT` 을 사용 (혹은 `ROLLBACK;`)
- `psycopg2` 의 트랜잭션
    - `autocommit` 이라는 파라미터로 조절가능
    - `autocommit=True` 가 되면, 기본적으로 `PostgreSQL` 의 커밋 모드와 동일
    - `autocommit=False` 가 되면, 커넥션 객체의 `.commit()` 과 `.rollbak()` 함수로 트랜잭션 조절 가능
    - 무엇을 사용할지는 개인 취향  

## DELETE FROM vs. TRUNCATE
- `DELETE FROM table_name (not DELETE * FROM)`
    - 테이블에서 모든 레코드를 삭제
    - vs. `DROP TABLE table_name`
    - `WHERE` 사용해 특정 레코드만 삭제 가능;
        - `DELETE FROM raw_data.user_session_channel WHERE channel = 'Google'`
- `TRUNCATE table_name` 도 테이블에서 모든 레코드를 삭제
    - `DELETE FROM` 의 subset이라고 볼 수 있음
    - `DELETE FROM` 속도가 느림
    - `TRUNCATE` 이 전체 테이블의 내용 삭제시에는 여로모로 유리
    - 하지만 두가지 단점이 존재
        - `TRUNCATE` 는 `WHERE` 을 지원하지 않음
        - **`TRUNCATE` 는 `Transaction` 을 지원하지 않음** 
- `DELETE FROM` 은 트랜잭션 안에서 사용 가능 `TRUNCATE` 은 트랜잭션 안에서 사용해도 `ROLLBACK` 이 안됨.

## 트랜잭션 실습
```sql
DROP TABLE IF EXISTS adhoc.keeyong_name_gender;
CREATE TABLE adhoc.keeyong_name_gender (
  name varchar(32),
  gender varchar(16)
);
INSERT INTO adhoc.keeyong_name_gender VALUES ('Ben', 'Male'), ('Maddie', 'Female');
```
- 구글 Colab 은 Redshift 을 연결할때 `autocommit=True` 임
- `INSERT INTO adhoc.keeyong_name_gender VALUES ('Ben', 'Male'), ('Maddie', 'Female');` 하면 바로 추가가 됨
- 결과
```sql
SELECT *
FROM adhoc.keeyong_name_gender;
-- name 	gender
-- Ben 	Male
-- Maddie 	Female
```  

- psycopg2: Python 에서 PostgreSQL 계열의 RDB 를 연결할때 사용하는 모듈임.
    - 
        ```py
        import psycopg2

        # Redshift connection 함수
        def get_Redshift_connection(autocommit):
            host = "learnde.cduaw970ssvt.ap-northeast-2.redshift.amazonaws.com"
            redshift_user = "guest"
            redshift_pass = "Guest1!*"
            port = 5439
            dbname = "dev"
            conn = psycopg2.connect("dbname={dbname} user={user} host={host} password={password} port={port}".format(
                dbname=dbname,
                user=redshift_user,
                password=redshift_pass,
                host=host,
                port=port
            ))
            conn.set_session(autocommit=autocommit)
            return conn
        ```
    - `autocommit`: True/False
    - `autocommit=False`로 했을 경우
- INSERT SQL을 autocommit=False로 실행하고 psycopg2로 컨트롤하기
    - 
        ```py
        try:
            cur.execute("DELETE FROM adhoc.keeyong_name_gender;") 
            cur.execute("INSERT INTO adhoc.keeyong_name_gender VALUES ('Claire', 'Female');")
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            conn.rollback()
        finally :
            conn.close()
        ```
- 잘못된 SQL을 중간에 실행해보기
    - 
        ```py
        cur.execute("BEGIN;")
        cur.execute("DELETE FROM adhoc.keeyong_name_gender;")
        cur.execute("INSERT INTO adhoc.keeyong_name_gender2 VALUES ('Andrew', 'Male');")
        cur.execute("END;")
        # ---------------------------------------------------------------------------

        # ProgrammingError                          Traceback (most recent call last)

        # <ipython-input-26-80c53e9acf2e> in <module>()
        #     1 cur.execute("BEGIN;")
        #     2 cur.execute("DELETE FROM adhoc.keeyong_name_gender;")
        # ----> 3 cur.execute("INSERT INTO adhoc.keeyong_name_gender2 VALUES ('Andrew', 'Male');")
        #     4 cur.execute("END;")

        # ProgrammingError: relation "adhoc.keeyong_name_gender2" does not exist
        ```



# 기타 고급 문법 소개와 실습
## 알아두면 유용한 SQL 문법들
- UNION, EXCEPT, INTERSECT
- COALESCE, NULLIF
- LISTAGG
- LAG
- WINDOW 함수
    - ROW_NUMBER OVER
    - SUM OVER
    - FIST_VALUE, LAST_VALUE
- JSON Parsing 함수  

## UNION, EXCEPT, INTERSECT
- UNION
    - 여러 개의 테이블들이나 SELECT 결과를 하나의 결과로 합쳐줌
    - `UNION` vs. `UNION ALL`
        - `UNION` 은 중복을 제거
        - `UNION ALL`: 중복되는 레코드가 있다고 해도 그대로 놔둠.
    - 
        ```sql
        SELECT 'keeyong' as first_name, 'han' as last_name

        UNION

        SELECT 'elon', 'musk'

        UNION

        SELECT 'keeyong', 'han

        -- first_name 	last_name
        -- elon 	musk
        -- keeyong 	han
        ```  
    - 
        ```sql
        SELECT 'keeyong' as first_name, 'han' as last_name

        UNION ALL

        SELECT 'elon', 'musk'

        UNION ALL

        SELECT 'keeyong', 'han

        -- first_name 	last_name
        -- keeyong 	han
        -- elon 	musk
        -- keeyong 	han
        ```  

- EXCEPT
    - 하나의 `SELECT` 결과에서 다른 `SELECT` 결과를 빼주는 것이 가능
- INTERSECT
    - 여러 개의 `SELECT` 문에서 같은 레코드들만 찾아줌  

## COALESCE, NULLIF
- `COALESCE(Expression1, Expression2, ...)`
    - 첫번째 Expression 부터 값이 NULL 이 아닌 것이 나오면 그 값을 리턴하고 모두 NULL 이면 NULL 을 리턴
    - NULL 값을 다른 값으로 바꾸고 싶을 때 사용
- `NULLIF(Expression1, Expression2, ...)`
    - Expression1 과 Expression2 의 값이 같으면 NULL 을 리턴  

## LISTAGG
- GROUP BY 에서 사용되는 Aggregate 함수 중의 하나
- 사용자 ID 별로 채널을 순서대로 리스트
- ts 순으로 channel을 list함
- 
```sql
SELECT
    userid,
    LISTAGG(channel) WITHIN GROUP (ORDER BY ts) channels
FROM raw_data.user_session_channel usc
JOIN raw_data.session_timestamp st ON usc.sessionid = st.sessionid
GROUP BY 1
LIMIT 10;
-- userid 	channels
-- 27 	YoutubeGoogleNaverFacebookFacebookGoogleGoogleFacebookOrganicOrganicOrganicOrganicFacebookNaverNaverOrganicGoogleInstagramGoogleNaverFacebookFacebookFacebookFacebookYoutubeGoogleFacebookOrganicNaverFacebookYoutubeFacebookInstagramYoutubeOrganicOrganicFacebookOrganicGoogleNaverFacebookNaverYoutubeGoogleFacebookOrganicOrganicFacebookFacebookGoogleYoutubeInstagramNaverOrganicGoogleInstagramNaverOrganicInstagramGoogleFacebookYoutubeFacebookInstagramNaverOrganicGoogleFacebookInstagramNaverNaverGoogleFacebookOrganicNaverFacebookOrganicNaverGoogleInstagramOrganicYoutubeOrganicFacebookInstagramInstagramGoogleOrganicInstagramOrganicYoutubeNaverInstagramInstagramNaverOrganicYoutubeNaverYoutubeInstagramNaverGoogleFacebookNaverFacebookOrganicOrganicYoutubeYoutubeYoutubeFacebookInstagramInstagramOrganicInstagramYoutubeYoutubeYoutubeInstagramNaverOrganicYoutubeGoogleYoutubeFacebookNaverInstagramGoogleGoogleGoogleOrganicOrganicOrganicOrganicInstagramInstagramOrganicFacebookYoutubeYoutubeFacebookYoutubeFacebookFacebookInstagramGoogleFacebookNaverOrganicInstagramFacebookInstagramGoogleFacebookYoutubeOrganicNaverGoogleInstagram
-- 29 	NaverYoutubeGoogleFacebookNaverGoogleOrganicOrganicGoogleOrganicInstagramOrganicFacebookGoogleNaverInstagramNaverYoutubeInstagramGoogleNaverYoutubeYoutubeOrganicOrganicOrganicNaverFacebookInstagramGoogleYoutubeGoogleFacebookInstagramNaverInstagramOrganicOrganicInstagramYoutubeInstagramFacebookGoogleGoogleInstagramYoutubeOrganicFacebookFacebookNaverInstagramOrganicInstagramNaverYoutubeInstagramYoutubeNaverYoutubeOrganicOrganicFacebookInstagramFacebookNaverInstagramNaverFacebookOrganicGoogleInstagramNaverInstagramInstagramYoutubeFacebookInstagramOrganicYoutubeOrganicOrganicNaver
-- 33 	GoogleInstagramInstagramInstagramOrganicNaverNaverNaverFacebookOrganicInstagramYoutubeNaverNaverInstagramYoutubeFacebookYoutubeInstagramOrganicNaverInstagramGoogleGoogleInstagramInstagramInstagramYoutubeInstagramGoogleFacebookFacebookGoogleGoogleYoutubeOrganicFacebookGoogleInstagramFacebookFacebookGoogleInstagramNaverInstagramFacebookOrganicInstagramYoutubeGoogleInstagramNaverNaverInstagramOrganicFacebookGoogleGoogleGoogleInstagramFacebookFacebookOrganicOrganicOrganicYoutubeInstagramYoutubeFacebookInstagramFacebookFacebookGoogleOrganicInstagramInstagramNaverInstagramOrganicYoutubeOrganicOrganicYoutubeOrganicYoutubeNaverOrganicYoutubeInstagramYoutubeInstagramYoutubeNaverGoogleOrganicNaverOrganicYoutubeInstagramYoutubeNaverOrganicOrganicYoutubeGoogleFacebookGoogleInstagramNaverFacebookInstagramGoogleFacebookYoutubeOrganicInstagramNaverOrganicOrganicOrganicOrganicOrganicInstagramNaverInstagramInstagramOrganicGoogleFacebookNaverGoogleInstagramInstagramNaverYoutubeOrganicInstagramNaverInstagramFacebookOrganicFacebookYoutubeInstagramYoutubeNaverInstagramInstagramYoutubeOrganicOrganicInstagramYoutubeGoogleGoogleNaverGoogleFacebookNaverYoutubeYoutubeYoutubeGoogleGoogleYoutubeNaverNaverYoutubeNaverNaverOrganicNaverFacebookYoutubeInstagramFacebookFacebookYoutubeGoogleOrganicFacebookInstagramOrganicNaverGoogleInstagramOrganicYoutubeYoutubeYoutubeYoutubeFacebookInstagramInstagramInstagramNaverGoogleNaverFacebookInstagramYoutubeFacebookNaverGoogleFacebookOrganicNaverYoutubeOrganicYoutubeYoutubeNaverYoutubeNaverFacebookGoogleFacebookInstagramGoogleOrganicNaverYoutube
-- 40 	YoutubeYoutubeNaverGoogleNaverNaverYoutubeYoutubeFacebookNaverYoutubeInstagramNaverOrganicOrganicOrganicYoutubeYoutubeGoogleGoogleNaverNaverYoutubeFacebookFacebookFacebookFacebookInstagramYoutubeGoogleOrganicNaverGoogleInstagramGoogleOrganicNaverInstagramFacebookInstagramInstagramNaverOrganicNaverOrganicGoogleYoutubeGoogleFacebookGoogleYoutubeOrganicInstagramInstagramYoutubeNaverOrganicInstagramInstagramInstagramNaverOrganicOrganicFacebookGoogleNaverYoutubeGoogleNaverOrganicGoogleOrganicYoutubeOrganicFacebookFacebookOrganicYoutubeInstagramInstagramNaverNaverYoutubeGoogleGoogleNaverGoogleFacebookInstagramGoogleFacebookNaverYoutubeGoogleOrganicYoutubeOrganicInstagramInstagramGoogleFacebookNaverGoogleGoogleOrganicNaverGoogleInstagramNaverFacebookInstagramFacebookNaverOrganicGoogleInstagramOrganicNaverOrganicNaverGoogleGoogleInstagramFacebookFacebookYoutubeFacebookOrganicNaverOrganicGoogleOrganicGoogleYoutubeOrganicNaverInstagramOrganicOrganicYoutubeYoutubeInstagramNaverFacebookFacebookGoogleOrganicOrganicInstagramGoogleGoogleYoutubeYoutubeGoogleYoutubeInstagramFacebookYoutubeFacebookGoogleInstagramInstagramOrganicFacebookNaverOrganicNaverNaverNaverFacebookGoogleOrganicFacebookGoogleGoogleInstagramFacebookYoutubeInstagramGoogleOrganicInstagramGoogleInstagramOrganicNaverYoutubeYoutubeGoogleYoutubeYoutubeYoutubeNaverNaverFacebookNaverFacebookYoutubeNaverOrganicGoogleFacebookInstagramFacebookOrganicFacebookOrganicGoogleYoutubeYoutubeOrganicFacebookOrganicFacebookGoogleInstagramOrganicNaverNaverFacebookNaverOrganicInstagramOrganicNaverGoogleYoutubeGoogleGoogleInstagramYoutubeFacebookInstagramInstagramNaverGoogleFacebookYoutubeNaverInstagramInstagramYoutubeYoutubeOrganicGoogle
-- 44 	NaverYoutubeFacebookYoutubeYoutubeGoogleNaverFacebookInstagramInstagramInstagramGoogleOrganicNaverFacebookYoutubeYoutubeInstagramOrganicNaverYoutubeYoutubeInstagramGoogleNaverGoogleNaverOrganicYoutubeInstagram
-- 59 	InstagramOrganicNaverInstagramGoogleYoutubeNaverFacebookYoutubeYoutubeYoutubeFacebookFacebookNaverInstagramOrganicInstagramOrganicFacebookYoutubeGoogleOrganicFacebookGoogleFacebookNaverYoutubeFacebookFacebookGoogleYoutubeYoutubeInstagramNaverNaverFacebookYoutubeGoogleGoogleOrganicGoogleOrganicYoutubeFacebookFacebookYoutubeFacebookYoutubeYoutubeInstagramNaverGoogleInstagramYoutubeInstagramGoogleGoogleGoogleOrganicNaverGoogleYoutubeNaverInstagramYoutubeGoogleOrganicInstagramInstagramGoogleOrganicFacebookNaverOrganicNaverNaverFacebookFacebookYoutubeNaverYoutubeFacebookInstagramGoogleNaverFacebookGoogleYoutubeNaverNaverNaverYoutubeGoogleNaverOrganicInstagramOrganicOrganicOrganicYoutubeYoutubeOrganicInstagramNaverFacebookNaverFacebookOrganicYoutubeFacebookOrganicFacebookFacebookYoutubeOrganicInstagramGoogleInstagramYoutubeOrganicNaverNaverYoutubeFacebookNaverNaverInstagramOrganicInstagramInstagramNaverInstagramGoogleNaverFacebookFacebookInstagramNaverYoutubeFacebookOrganicInstagramNaverNaverInstagramNaverGoogleNaverFacebookInstagramGoogleNaverYoutubeNaverNaverGoogleGoogleOrganicGoogleInstagram
-- 68 	YoutubeGoogleInstagramYoutubeInstagramInstagramInstagramOrganicInstagramYoutubeGoogleGoogleOrganic
-- 87 	YoutubeYoutubeFacebookNaverFacebookYoutubeInstagramNaverFacebookGoogleYoutubeFacebookYoutubeYoutubeFacebookYoutubeYoutubeGoogleFacebookYoutubeInstagramInstagramInstagramFacebookYoutubeFacebookInstagramInstagramYoutubeYoutubeInstagramFacebookFacebookOrganicYoutubeOrganicFacebookYoutubeYoutubeFacebookInstagramYoutubeInstagramYoutubeOrganicOrganicYoutubeOrganicYoutubeYoutubeInstagramOrganicNaverFacebookInstagramFacebookYoutubeOrganicNaverNaverGoogleYoutubeOrganicYoutubeNaverNaverOrganicYoutubeInstagramFacebookGoogleOrganicYoutubeNaverGoogleNaverNaverFacebookGoogleYoutubeOrganicInstagramGoogleFacebookNaverOrganicOrganicNaverOrganicGoogleNaverYoutubeNaverYoutubeGoogleInstagramOrganicGoogleGoogle
-- 97 	OrganicFacebookInstagramOrganicOrganicNaverInstagramFacebookYoutubeInstagramOrganicGoogleFacebookNaverYoutubeGoogleInstagramFacebookGoogleInstagramGoogleNaverYoutubeNaverFacebookInstagramNaverNaverGoogleGoogleFacebookGoogleOrganicOrganicGoogleInstagramFacebookFacebookInstagramGoogleInstagramOrganicInstagramGoogleInstagramFacebookYoutubeNaverGoogleInstagramFacebookFacebookFacebookOrganicNaverFacebookYoutubeYoutubeNaverInstagramFacebookFacebookYoutubeOrganicYoutubeYoutubeInstagramGoogleNaverOrganicNaverNaverYoutubeYoutubeGoogleFacebookOrganicFacebookNaverFacebookOrganicYoutubeOrganicNaverInstagramFacebookYoutubeFacebookInstagramGoogleYoutubeYoutubeFacebookFacebookGoogleInstagramOrganicYoutubeNaverInstagramGoogleFacebookFacebookGoogleGoogleNaverFacebookYoutubeOrganicYoutubeFacebookFacebookYoutubeOrganicNaverNaverGoogleOrganicYoutubeNaverFacebookOrganicYoutubeNaverNaverFacebookNaverFacebookNaverFacebookNaverNaverInstagramNaverOrganicFacebookOrganicOrganicGoogleInstagramNaverNaverFacebookNaverInstagramNaverNaverNaverFacebookNaverGoogleOrganicOrganicYoutubeYoutubeInstagramInstagramNaverGoogleYoutubeInstagramYoutubeInstagramNaverFacebookYoutubeInstagramFacebookOrganicGoogleInstagramGoogleYoutubeFacebookOrganic
-- 113 	OrganicOrganicOrganicOrganicInstagramGoogleFacebookYoutubeOrganicFacebookNaverOrganicFacebookFacebookYoutubeOrganicNaverInstagramGoogleFacebookYoutubeNaverFacebookOrganicNaverYoutubeNaverFacebookNaverYoutubeYoutubeOrganicFacebookNaverFacebookOrganicGoogleInstagramInstagramInstagramInstagramFacebookOrganicOrganicOrganicOrganicYoutubeOrganicGoogleFacebookInstagramInstagramGoogleInstagramInstagramInstagramOrganicGoogleInstagramFacebookFacebookInstagramYoutubeYoutubeOrganicInstagramFacebookGoogleYoutubeOrganicGoogleYoutubeOrganicInstagramYoutubeOrganicGoogleGoogleOrganicFacebookGoogleNaverNaverYoutubeNaverNaverInstagramFacebookFacebookYoutubeYoutubeInstagramYoutubeYoutubeInstagramFacebookGoogleYoutubeOrganicYoutubeOrganicNaverFacebookYoutubeOrganicNaverNaverNaverNaverYoutubeGoogleYoutubeFacebookOrganicOrganicGoogleGoogleNaverGoogleYoutubeNaverInstagramYoutubeFacebookOrganicOrganicGoogleYoutubeInstagramNaverNaverGoogleInstagramFacebookGoogleFacebookOrganicGoogleOrganicYoutubeNaverFacebookInstagramOrganicOrganicFacebookFacebookFacebookOrganicYoutubeInstagramOrganicNaverOrganicFacebookGoogleOrganicOrganicGoogleFacebookNaverNaverNaverFacebookOrganicOrganicNaverInstagramYoutubeInstagramInstagramInstagramYoutubeOrganicNaverGoogleOrganicOrganicYoutubeOrganicFacebookYoutubeYoutubeFacebookInstagramNaverNaverYoutubeOrganicGoogleInstagramYoutubeFacebookNaverNaverOrganicGoogleNaverNaverNaverOrganicFacebookOrganicGoogleNaverFacebookInstagramNaverGoogleNaverGoogleFacebookYoutubeOrganicOrganic
```
- 
```sql
SELECT
    userid,
    LISTAGG(channel, '->') WITHIN GROUP (ORDER BY ts) channels
FROM raw_data.user_session_channel usc
JOIN raw_data.session_timestamp st ON usc.sessionid = st.sessionid
GROUP BY 1
LIMIT 10;
-- userid 	channels
-- 27 	Youtube->Google->Naver->Facebook->Facebook->Google->Google->Facebook->Organic->Organic->Organic->Organic->Facebook->Naver->Naver->Organic->Google->Instagram->Google->Naver->Facebook->Facebook->Facebook->Facebook->Youtube->Google->Facebook->Organic->Naver->Facebook->Youtube->Facebook->Instagram->Youtube->Organic->Organic->Facebook->Organic->Google->Naver->Facebook->Naver->Youtube->Google->Facebook->Organic->Organic->Facebook->Facebook->Google->Youtube->Instagram->Naver->Organic->Google->Instagram->Naver->Organic->Instagram->Google->Facebook->Youtube->Facebook->Instagram->Naver->Organic->Google->Facebook->Instagram->Naver->Naver->Google->Facebook->Organic->Naver->Facebook->Organic->Naver->Google->Instagram->Organic->Youtube->Organic->Facebook->Instagram->Instagram->Google->Organic->Instagram->Organic->Youtube->Naver->Instagram->Instagram->Naver->Organic->Youtube->Naver->Youtube->Instagram->Naver->Google->Facebook->Naver->Facebook->Organic->Organic->Youtube->Youtube->Youtube->Facebook->Instagram->Instagram->Organic->Instagram->Youtube->Youtube->Youtube->Instagram->Naver->Organic->Youtube->Google->Youtube->Facebook->Naver->Instagram->Google->Google->Google->Organic->Organic->Organic->Organic->Instagram->Instagram->Organic->Facebook->Youtube->Youtube->Facebook->Youtube->Facebook->Facebook->Instagram->Google->Facebook->Naver->Organic->Instagram->Facebook->Instagram->Google->Facebook->Youtube->Organic->Naver->Google->Instagram
-- 29 	Naver->Youtube->Google->Facebook->Naver->Google->Organic->Organic->Google->Organic->Instagram->Organic->Facebook->Google->Naver->Instagram->Naver->Youtube->Instagram->Google->Naver->Youtube->Youtube->Organic->Organic->Organic->Naver->Facebook->Instagram->Google->Youtube->Google->Facebook->Instagram->Naver->Instagram->Organic->Organic->Instagram->Youtube->Instagram->Facebook->Google->Google->Instagram->Youtube->Organic->Facebook->Facebook->Naver->Instagram->Organic->Instagram->Naver->Youtube->Instagram->Youtube->Naver->Youtube->Organic->Organic->Facebook->Instagram->Facebook->Naver->Instagram->Naver->Facebook->Organic->Google->Instagram->Naver->Instagram->Instagram->Youtube->Facebook->Instagram->Organic->Youtube->Organic->Organic->Naver
-- 33 	Google->Instagram->Instagram->Instagram->Organic->Naver->Naver->Naver->Facebook->Organic->Instagram->Youtube->Naver->Naver->Instagram->Youtube->Facebook->Youtube->Instagram->Organic->Naver->Instagram->Google->Google->Instagram->Instagram->Instagram->Youtube->Instagram->Google->Facebook->Facebook->Google->Google->Youtube->Organic->Facebook->Google->Instagram->Facebook->Facebook->Google->Instagram->Naver->Instagram->Facebook->Organic->Instagram->Youtube->Google->Instagram->Naver->Naver->Instagram->Organic->Facebook->Google->Google->Google->Instagram->Facebook->Facebook->Organic->Organic->Organic->Youtube->Instagram->Youtube->Facebook->Instagram->Facebook->Facebook->Google->Organic->Instagram->Instagram->Naver->Instagram->Organic->Youtube->Organic->Organic->Youtube->Organic->Youtube->Naver->Organic->Youtube->Instagram->Youtube->Instagram->Youtube->Naver->Google->Organic->Naver->Organic->Youtube->Instagram->Youtube->Naver->Organic->Organic->Youtube->Google->Facebook->Google->Instagram->Naver->Facebook->Instagram->Google->Facebook->Youtube->Organic->Instagram->Naver->Organic->Organic->Organic->Organic->Organic->Instagram->Naver->Instagram->Instagram->Organic->Google->Facebook->Naver->Google->Instagram->Instagram->Naver->Youtube->Organic->Instagram->Naver->Instagram->Facebook->Organic->Facebook->Youtube->Instagram->Youtube->Naver->Instagram->Instagram->Youtube->Organic->Organic->Instagram->Youtube->Google->Google->Naver->Google->Facebook->Naver->Youtube->Youtube->Youtube->Google->Google->Youtube->Naver->Naver->Youtube->Naver->Naver->Organic->Naver->Facebook->Youtube->Instagram->Facebook->Facebook->Youtube->Google->Organic->Facebook->Instagram->Organic->Naver->Google->Instagram->Organic->Youtube->Youtube->Youtube->Youtube->Facebook->Instagram->Instagram->Instagram->Naver->Google->Naver->Facebook->Instagram->Youtube->Facebook->Naver->Google->Facebook->Organic->Naver->Youtube->Organic->Youtube->Youtube->Naver->Youtube->Naver->Facebook->Google->Facebook->Instagram->Google->Organic->Naver->Youtube
-- 40 	Youtube->Youtube->Naver->Google->Naver->Naver->Youtube->Youtube->Facebook->Naver->Youtube->Instagram->Naver->Organic->Organic->Organic->Youtube->Youtube->Google->Google->Naver->Naver->Youtube->Facebook->Facebook->Facebook->Facebook->Instagram->Youtube->Google->Organic->Naver->Google->Instagram->Google->Organic->Naver->Instagram->Facebook->Instagram->Instagram->Naver->Organic->Naver->Organic->Google->Youtube->Google->Facebook->Google->Youtube->Organic->Instagram->Instagram->Youtube->Naver->Organic->Instagram->Instagram->Instagram->Naver->Organic->Organic->Facebook->Google->Naver->Youtube->Google->Naver->Organic->Google->Organic->Youtube->Organic->Facebook->Facebook->Organic->Youtube->Instagram->Instagram->Naver->Naver->Youtube->Google->Google->Naver->Google->Facebook->Instagram->Google->Facebook->Naver->Youtube->Google->Organic->Youtube->Organic->Instagram->Instagram->Google->Facebook->Naver->Google->Google->Organic->Naver->Google->Instagram->Naver->Facebook->Instagram->Facebook->Naver->Organic->Google->Instagram->Organic->Naver->Organic->Naver->Google->Google->Instagram->Facebook->Facebook->Youtube->Facebook->Organic->Naver->Organic->Google->Organic->Google->Youtube->Organic->Naver->Instagram->Organic->Organic->Youtube->Youtube->Instagram->Naver->Facebook->Facebook->Google->Organic->Organic->Instagram->Google->Google->Youtube->Youtube->Google->Youtube->Instagram->Facebook->Youtube->Facebook->Google->Instagram->Instagram->Organic->Facebook->Naver->Organic->Naver->Naver->Naver->Facebook->Google->Organic->Facebook->Google->Google->Instagram->Facebook->Youtube->Instagram->Google->Organic->Instagram->Google->Instagram->Organic->Naver->Youtube->Youtube->Google->Youtube->Youtube->Youtube->Naver->Naver->Facebook->Naver->Facebook->Youtube->Naver->Organic->Google->Facebook->Instagram->Facebook->Organic->Facebook->Organic->Google->Youtube->Youtube->Organic->Facebook->Organic->Facebook->Google->Instagram->Organic->Naver->Naver->Facebook->Naver->Organic->Instagram->Organic->Naver->Google->Youtube->Google->Google->Instagram->Youtube->Facebook->Instagram->Instagram->Naver->Google->Facebook->Youtube->Naver->Instagram->Instagram->Youtube->Youtube->Organic->Google
-- 44 	Naver->Youtube->Facebook->Youtube->Youtube->Google->Naver->Facebook->Instagram->Instagram->Instagram->Google->Organic->Naver->Facebook->Youtube->Youtube->Instagram->Organic->Naver->Youtube->Youtube->Instagram->Google->Naver->Google->Naver->Organic->Youtube->Instagram
-- 59 	Instagram->Organic->Naver->Instagram->Google->Youtube->Naver->Facebook->Youtube->Youtube->Youtube->Facebook->Facebook->Naver->Instagram->Organic->Instagram->Organic->Facebook->Youtube->Google->Organic->Facebook->Google->Facebook->Naver->Youtube->Facebook->Facebook->Google->Youtube->Youtube->Instagram->Naver->Naver->Facebook->Youtube->Google->Google->Organic->Google->Organic->Youtube->Facebook->Facebook->Youtube->Facebook->Youtube->Youtube->Instagram->Naver->Google->Instagram->Youtube->Instagram->Google->Google->Google->Organic->Naver->Google->Youtube->Naver->Instagram->Youtube->Google->Organic->Instagram->Instagram->Google->Organic->Facebook->Naver->Organic->Naver->Naver->Facebook->Facebook->Youtube->Naver->Youtube->Facebook->Instagram->Google->Naver->Facebook->Google->Youtube->Naver->Naver->Naver->Youtube->Google->Naver->Organic->Instagram->Organic->Organic->Organic->Youtube->Youtube->Organic->Instagram->Naver->Facebook->Naver->Facebook->Organic->Youtube->Facebook->Organic->Facebook->Facebook->Youtube->Organic->Instagram->Google->Instagram->Youtube->Organic->Naver->Naver->Youtube->Facebook->Naver->Naver->Instagram->Organic->Instagram->Instagram->Naver->Instagram->Google->Naver->Facebook->Facebook->Instagram->Naver->Youtube->Facebook->Organic->Instagram->Naver->Naver->Instagram->Naver->Google->Naver->Facebook->Instagram->Google->Naver->Youtube->Naver->Naver->Google->Google->Organic->Google->Instagram
-- 68 	Youtube->Google->Instagram->Youtube->Instagram->Instagram->Instagram->Organic->Instagram->Youtube->Google->Google->Organic
-- 87 	Youtube->Youtube->Facebook->Naver->Facebook->Youtube->Instagram->Naver->Facebook->Google->Youtube->Facebook->Youtube->Youtube->Facebook->Youtube->Youtube->Google->Facebook->Youtube->Instagram->Instagram->Instagram->Facebook->Youtube->Facebook->Instagram->Instagram->Youtube->Youtube->Instagram->Facebook->Facebook->Organic->Youtube->Organic->Facebook->Youtube->Youtube->Facebook->Instagram->Youtube->Instagram->Youtube->Organic->Organic->Youtube->Organic->Youtube->Youtube->Instagram->Organic->Naver->Facebook->Instagram->Facebook->Youtube->Organic->Naver->Naver->Google->Youtube->Organic->Youtube->Naver->Naver->Organic->Youtube->Instagram->Facebook->Google->Organic->Youtube->Naver->Google->Naver->Naver->Facebook->Google->Youtube->Organic->Instagram->Google->Facebook->Naver->Organic->Organic->Naver->Organic->Google->Naver->Youtube->Naver->Youtube->Google->Instagram->Organic->Google->Google
-- 97 	Organic->Facebook->Instagram->Organic->Organic->Naver->Instagram->Facebook->Youtube->Instagram->Organic->Google->Facebook->Naver->Youtube->Google->Instagram->Facebook->Google->Instagram->Google->Naver->Youtube->Naver->Facebook->Instagram->Naver->Naver->Google->Google->Facebook->Google->Organic->Organic->Google->Instagram->Facebook->Facebook->Instagram->Google->Instagram->Organic->Instagram->Google->Instagram->Facebook->Youtube->Naver->Google->Instagram->Facebook->Facebook->Facebook->Organic->Naver->Facebook->Youtube->Youtube->Naver->Instagram->Facebook->Facebook->Youtube->Organic->Youtube->Youtube->Instagram->Google->Naver->Organic->Naver->Naver->Youtube->Youtube->Google->Facebook->Organic->Facebook->Naver->Facebook->Organic->Youtube->Organic->Naver->Instagram->Facebook->Youtube->Facebook->Instagram->Google->Youtube->Youtube->Facebook->Facebook->Google->Instagram->Organic->Youtube->Naver->Instagram->Google->Facebook->Facebook->Google->Google->Naver->Facebook->Youtube->Organic->Youtube->Facebook->Facebook->Youtube->Organic->Naver->Naver->Google->Organic->Youtube->Naver->Facebook->Organic->Youtube->Naver->Naver->Facebook->Naver->Facebook->Naver->Facebook->Naver->Naver->Instagram->Naver->Organic->Facebook->Organic->Organic->Google->Instagram->Naver->Naver->Facebook->Naver->Instagram->Naver->Naver->Naver->Facebook->Naver->Google->Organic->Organic->Youtube->Youtube->Instagram->Instagram->Naver->Google->Youtube->Instagram->Youtube->Instagram->Naver->Facebook->Youtube->Instagram->Facebook->Organic->Google->Instagram->Google->Youtube->Facebook->Organic
-- 113 	Organic->Organic->Organic->Organic->Instagram->Google->Facebook->Youtube->Organic->Facebook->Naver->Organic->Facebook->Facebook->Youtube->Organic->Naver->Instagram->Google->Facebook->Youtube->Naver->Facebook->Organic->Naver->Youtube->Naver->Facebook->Naver->Youtube->Youtube->Organic->Facebook->Naver->Facebook->Organic->Google->Instagram->Instagram->Instagram->Instagram->Facebook->Organic->Organic->Organic->Organic->Youtube->Organic->Google->Facebook->Instagram->Instagram->Google->Instagram->Instagram->Instagram->Organic->Google->Instagram->Facebook->Facebook->Instagram->Youtube->Youtube->Organic->Instagram->Facebook->Google->Youtube->Organic->Google->Youtube->Organic->Instagram->Youtube->Organic->Google->Google->Organic->Facebook->Google->Naver->Naver->Youtube->Naver->Naver->Instagram->Facebook->Facebook->Youtube->Youtube->Instagram->Youtube->Youtube->Instagram->Facebook->Google->Youtube->Organic->Youtube->Organic->Naver->Facebook->Youtube->Organic->Naver->Naver->Naver->Naver->Youtube->Google->Youtube->Facebook->Organic->Organic->Google->Google->Naver->Google->Youtube->Naver->Instagram->Youtube->Facebook->Organic->Organic->Google->Youtube->Instagram->Naver->Naver->Google->Instagram->Facebook->Google->Facebook->Organic->Google->Organic->Youtube->Naver->Facebook->Instagram->Organic->Organic->Facebook->Facebook->Facebook->Organic->Youtube->Instagram->Organic->Naver->Organic->Facebook->Google->Organic->Organic->Google->Facebook->Naver->Naver->Naver->Facebook->Organic->Organic->Naver->Instagram->Youtube->Instagram->Instagram->Instagram->Youtube->Organic->Naver->Google->Organic->Organic->Youtube->Organic->Facebook->Youtube->Youtube->Facebook->Instagram->Naver->Naver->Youtube->Organic->Google->Instagram->Youtube->Facebook->Naver->Naver->Organic->Google->Naver->Naver->Naver->Organic->Facebook->Organic->Google->Naver->Facebook->Instagram->Naver->Google->Naver->Google->Facebook->Youtube->Organic->Organic
```  

## WINDOW
- Syntax
    - function(expression) OVER ( [**PARTITION BY** expression] [**ORDER BY** expression] )
- **Useful functions**
    - **ROW_NUMBER**, FIRST_VALUE, LAST_VALUE, LAG
    - Math functions: AVG, SUM, COUNT, MAX, MIN, MEDIAN, NTH_VALUE
- 어떤 사용자 세션에서 시간순으로 봤을 때
    - 앞 세션의 채널이 무엇인지 알고 싶다면
    - 혹은 다음 세션의 채널이 무엇인지 알고 싶다면
- 
```sql
-- 이전 채널 찾기
SELECT usc.*, st.ts,
    LAG(channel, 1) OVER (PARTITION BY userId ORDER BY ts) prev_channel
FROM raw_data.user_session_channel usc
JOIN raw_data.session_timestamp st ON usc.sessionid = st.sessionid
ORDER BY usc.userid, st.ts
-- 100 rows affected.
-- userid 	sessionid 	channel 	ts 	prev_channel
-- 27 	a67c8c9a961b4182688768dd9ba015fe 	Youtube 	2019-05-01 17:03:59.957000 	Google
-- 27 	b04c387c8384ca083a71b8da516f65f6 	Google 	2019-05-02 19:21:30.280000 	Naver
-- 27 	abebb7c39f4b5e46bbcfab2b565ef32b 	Naver 	2019-05-03 20:38:40.730000 	Facebook
-- 27 	ab49ef78e2877bfd2c2bfa738e459bf0 	Facebook 	2019-05-04 21:48:07.300000 	Facebook
-- 27 	f740c8d9c193f16d8a07d3a8a751d13f 	Facebook 	2019-05-05 18:15:30.540000 	Google
-- 27 	ef452c63f81d0105dd4486f775adec81 	Google 	2019-05-06 17:49:15.437000 	Google
-- 27 	d7f426ccbc6db7e235c57958c21c5dfa 	Google 	2019-05-07 20:41:25.293000 	Facebook
-- 27 	df334b223e699294764c2bb7ae40d8db 	Facebook 	2019-05-08 17:35:16.440000 	Organic
```  
- 다음 채널을 찾으려면?
    - 
    ```sql
    SELECT usc.*, st.ts, LAG(channel, 1) OVER (PARTITION BY userId ORDER BY ts DESC) prev_channel 
    FROM raw_data.user_session_channel usc
    JOIN raw_data.session_timestamp st ON usc.sessionid = st.sessionid
    ORDER BY usc.userid, st.ts
    LIMIT 100;
    --userid 	sessionid 	channel 	ts 	prev_channel
    --27 	a67c8c9a961b4182688768dd9ba015fe 	Youtube 	2019-05-01 17:03:59.957000 	Google
    --27 	b04c387c8384ca083a71b8da516f65f6 	Google 	2019-05-02 19:21:30.280000 	Naver
    --27 	abebb7c39f4b5e46bbcfab2b565ef32b 	Naver 	2019-05-03 20:38:40.730000 	Facebook
    --27 	ab49ef78e2877bfd2c2bfa738e459bf0 	Facebook 	2019-05-04 21:48:07.300000 	Facebook
    --27 	f740c8d9c193f16d8a07d3a8a751d13f 	Facebook 	2019-05-05 18:15:30.540000 	Google
    --27 	ef452c63f81d0105dd4486f775adec81 	Google 	2019-05-06 17:49:15.437000 	Google
    --27 	d7f426ccbc6db7e235c57958c21c5dfa 	Google 	2019-05-07 20:41:25.293000 	Facebook
    ```

## JSON Parsing
- **JSON 의 포맷을 이미 아는 상황에서만** 사용가능한 함수
    - JSON string 을 입력으로 받아 특정 필드의 값을 추출가능 (nested 구조 지원)
- JSON_EXTRACT_PATH_NEXT
    - 
    ```sql
    SELECT JSON_EXTRACT_PATH_TEXT('{"f2":{"f3": "1"},"f4":{"f5":"99","f6":"star"}}','f4', 'f6');
    -- json_extract_path_text
    -- star
    ```
    - 
    ```sql
    SELECT JSON_EXTRACT_PATH_TEXT('{"f2":{"f3": "1"},"f4":{"f5":"99","f6":"star"}}','f4');
    --json_extract_path_text
    --{"f5":"99","f6":"star"}
    ```

# 맺음말
