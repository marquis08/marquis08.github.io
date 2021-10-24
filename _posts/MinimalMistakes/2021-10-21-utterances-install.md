---
date: 2021-10-21 14:12
title: "Utterances install in github blog"
categories: MinimalMistakes
tags: MinimalMistakes
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

Utterances 설치해서 댓글 기능 추가하기

아래 출처 블로그 참고 했다.

하지만 언급안한 부분이 있는데 (너무 당연하기도 하고..)

`_config.yml` 에서 맨아래 default 부분에서 `comments: true` 인지 확인해줘야 한다.  

예전에 초기에 default false로 해놔서 그런지 이 부분 디버깅 하는데 좀 시간이 걸렸다.  

당연하게도 로컬에서 디버깅할때는 표시가 되기 않는다.  


> <https://ansohxxn.github.io/blog/utterances/>