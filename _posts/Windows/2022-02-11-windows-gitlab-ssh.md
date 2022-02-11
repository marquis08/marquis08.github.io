---
date: 2022-02-11 11:00
title: "Windows gitlab ssh"
categories: gitlab ssh windows
tags: gitlab ssh windows
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

1. Execute git bash by windows key
2. command `ssh-keygen`
   1. Enter file in which to save the key (생성위치 정하는 파트)
      1. 기본경로로 할 예정이므로 ENTER
   2. Enter passphase
      1. 매번 push할 때마다 password를 입력하고 싶다면 여기서 입력한 PW로 매번 입력해줘야한다.
      2. push 할때마다 편하게 하려면 그냥 ENTER
3. 2번에서 생성한 위치에 public key가 생성되어 있는지 확인
4. ms publisher document로 안열린다.
   1. 연결 프로그램으로 메모장 선택후 열면 pub key(id_rsa.pub)를 확인할 수 있다.
5. GitLab에 SSH Key 등록하기
   1. Profile(오른쪽 상단 계정) &rarr; Preferences &rarr; SSH Keys
   2. 4번에서 열었던 pub key를 key에 복붙하면 된다.
      1. 개행될 경우 지워준다.
   3. Title은 편한것으로 하거나 아니면 자동으로 생성해준다.
   4. Expires at 을 따로 입력하지 않는다면 Never로 설정된다.
   5. `Add key`
6. 작업하던 repo로 들어가서 push를 해서 gitlab에 push되는지 확인한다.



# Appendix
## Reference
> gitlab ssh windows: <https://wylee-developer.tistory.com/54> 
>  
> 