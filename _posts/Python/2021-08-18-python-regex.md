---
date: 2021-08-18 14:52
title: "Python Regex"
categories: Regex Python
tags: Regex Python
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Regex
## 자주 사용하는 문자 클래스

- [0-9] 또는 [a-zA-Z] 등은 무척 자주 사용하는 정규 표현식이다. 이렇게 자주 사용하는 정규식은 별도의 표기법으로 표현할 수 있다. 다음을 기억해 두자.

- \d - 숫자와 매치, [0-9]와 동일한 표현식이다.
- \D - 숫자가 아닌 것과 매치, [^0-9]와 동일한 표현식이다.
- \s - whitespace 문자와 매치, [ \t\n\r\f\v]와 동일한 표현식이다. 맨 앞의 빈 칸은 공백문자(space)를 의미한다.
- \S - whitespace 문자가 아닌 것과 매치, [^ \t\n\r\f\v]와 동일한 표현식이다.
- \w - 문자+숫자(alphanumeric)와 매치, [a-zA-Z0-9_]와 동일한 표현식이다.
- \W - 문자+숫자(alphanumeric)가 아닌 문자와 매치, [^a-zA-Z0-9_]와 동일한 표현식이다.

## Raw String
- Python raw string is created by prefixing a string literal with `'r' or 'R'`. Python raw string treats backslash (`\`) as **a literal character**. This is useful when we want to have a string that contains backslash and don’t want it to be treated as an escape character.
- 특수 문자를 변경하지 않고 모든 character를 raw한 상태로 사용.
```py
s = 'Hi\nHello'
print(s)
### Output
# Hi
# Hello
```
```py
raw_s = r'Hi\nHello'
print(raw_s)
### Output
# Hi\nHello
```



# Appendix
## Reference
> Regular Expression HOWTO: <https://docs.python.org/3/howto/regex.html>  
> 점프 투 파이썬 07장 정규표현식 07-2 정규 표현식 시작하기: <https://wikidocs.net/4308>  
> python regex metacharacters: <https://www.geeksforgeeks.org/python-regex-metacharacters/>  
> Raw String: <https://www.journaldev.com/23598/python-raw-string>  
> Raw String: <https://frhyme.github.io/python/python_raw_string/>  