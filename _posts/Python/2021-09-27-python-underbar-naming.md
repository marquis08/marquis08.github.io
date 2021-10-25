---
date: 2021-09-27 14:30
title: "Python underbar underscore naminig"
categories: Python
tags: Python
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

4-1. 언더바가 앞에 하나 붙은 경우 (ex: _variable)
- 이 경우는 모듈 내에서만 해당 변수/함수를 사용하겠다는 의미입니다.(private) 특히 협업을 할 때 다른 팀원에게 '이 변수/함수는 이 모듈 내부에서만 사용할 거다'라고 명시적으로 힌트를 줄 수 있습니다.

단, 완전한 의미의 private는 아니기 때문에 여전히 해당 변수/함수는 접근하거나 사용할 수 있습니다. 파이썬은 private와 public의 확실한 구분이 없습니다.

하지만 외부 모듈에서 해당 모듈을 from module_name import * 식으로 전체 임포트 할 때 앞에 언더바가 붙은 변수/함수는 임포트를 하지 않습니다.
```py
# test_module.py

def _hi():
    print('hi')

def hello():
    print('hello')
```
test_module이라는 모듈을 만들었습니다. 그리고 함수를 2개 선언했는데 하나는 앞에 언더바를 붙이고 하나는 붙이지 않았습니다.
```py
from test_module import *

hello()
_hi() # 에러 발생
```
<https://tibetsandfox.tistory.com/20>

4-2. 뒤에 언더바가 하나 붙은 경우 (ex: variable_)
- 이 경우는 파이썬 키워드와 변수/함수명의 충돌을 피하기 위해 사용하는 경우입니다.

예를 들어, list나 print같은 키워드를 변수/함수명으로 사용하고 싶을 때 list_, print_ 와 같이 사용합니다.
```py
list_ = [1, 2, 3, 4, 5]

def print_(args):
    print('hi')
    print(args)

print_(list_)

출력 결과
hi
[1, 2, 3, 4, 5]
```