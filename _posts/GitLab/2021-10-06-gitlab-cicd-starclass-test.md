---
date: 2021-10-06 14:00
title: "GitLab cicd starclass test"
categories: GitLab cicd
tags: GitLab cicd
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---


1. make `test_run.py` to pass with python uniitest  
1-1. make cicd variables in `settings`.
<img src="../assets/images/cicd_variable.png" width="50%" height="50%">  

```yml
script:
...
--assets $FRIDA_TEST_ASSET_0 
--assets $FRIDA_TEST_ASSET_
```
with `parser.add_argument('--assets', default=[], action='append')` this.

the yml file means that append `CICD variable(FRIDA_TEST_ASSET_0 )` to `assets`




2. make `run.py` to pass pipeline test stage

3. remove redundancy not linked to build stage


# Artifact

# CICD_variables
- protected는 protected tag or protected branch 가 들어간 파이프라인에서만 접근이 가능.
- docs 다시 체크해서 정리

# masked variable
- 접근 권한 범위 설정과 관련있는 설정
- 파이프라인 실행 로그만 볼 수만 있는 사람(권한)이 masked 없으면 변수가 표시가됨
- masked(************)로 출력이 됨.

# Credential Vault
같은 서비스 도 있음.


# protected tags
- settings/repository 설정에 protected tags, protected branchs 에서 설정.
- 생성, 삭제 권한 없음
- GITLAB에서는 삭제가 안되게 하려고(플랫폼마다 기능이 다름 , 권한 범위도 다름)
- Wildcards(*) 를 사용하면 자동으로 dropdown이 생김.


# CI_COMMIT_TAG
- CI_commit_tag는 predefined variable인데 여기에 값이 써지는 경우는 해당 태그가 variable에 입력이 된 상태에서 진행
    - repository에 있는 tags 가 CO_COMMIT

- ONLYTAGS
    - TAG를 만들면 그 떄 TRIGGER가 되서 PIPELINE  실행이 됨
    - PROTECTED 여부 상관없이 PIPELINE 실행이 됨.
    - local 에서 작업후 push 해도 pipeline 실행이 안됨
    - tag가 생성될 때만 pipeline 실행.

- EXCLUDE 옵션이 있음
    - REGEX를 써서 해당 태그가 매칭이 되면 안함(PIPELINE)

- 특정 TAG를 가진 REPOSITORY에서 BUILD 하려면?
    - 새 tag를 그 과거 tag를 선택해서 새로 만들면 됨
    - 완전 다른 작업을 할때나.
        - 빌드 환경이 새 버전이 나오면서 바뀌거나 했을 경우 터짐
        - 예전 환경을 다시 가져와서 실행테스트 해볼 경우 (디버깅)

# TAG는 CICD 끝나고 다는 것이 진정한 CICD다

# CI
각자 만든걸 통합
# CD
통합후 DEPLOY






# Appendix
## argparse
- store_true
store_true 를 주게 되면 default 값으로 Namespace에 True가 저장된다.
```py
parser = argparse.ArgumentParser()
parser.add_argument('--foo', action='store_true')
parser.add_argument('--bar', action='store_false')
parser.add_argument('--baz', action='store_false')
parser.parse_args('--foo --bar'.split())
# Namespace(foo=True, bar=False, baz=True)
```

- append
```py
parser = argparse.ArgumentParser()
parser.add_argument('--foo', action='append')
parser.parse_args('--foo 1 --foo 2'.split())
# Namespace(foo=['1', '2'])
```


<https://docs.python.org/3/library/argparse.html>