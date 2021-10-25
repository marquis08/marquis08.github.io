---
date: 2021-10-03 01:30
title: "GitLab runner"
categories: GitLab
tags: GitLab
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---


bitbucket 용으로 ssh를 하나만 쓰다가

개인 gitlab용으로 ssh가 필요하거나 그럴 경우 

혹은 여러개의 계정이 ssh를 사용할 경우 필요하다.


1. 키생성
```sh
ssh-keygen -t rsa -b 4096 -C "your@email"
```

1-1. 키 이름 지정
Enter file in which to save the key(ssh_path): {your_new_key_name}

1-2. 저장된 pub, private key 확인
cat id_rsa_my

2. 키 추가 및 저장
```sh
ssh-add ~/.ssh/id_rsa_me
ssh-add ~/.ssh/id_rsa_work
ssh-add -l # 저장
```
2-1. 저장하려니 뜨는 에러 (ssh-add 해두면 비빌번호 매번 입력안해도 된다.)
`Could not open a connection to your authentication agent.`
passphrase에서 비밀번호를 입력해두었더니 뜨는 에러

`eval "$(ssh-agent -s)"`로 ssh-agent 백그라운드로 실행

```sh
eval "$(ssh-agent -s)" 
ssh-add ~/.ssh/id_rsa_gitlab
# Enter passphrase for /home/**********/.ssh/id_rsa_gitlab:
```


3. ssh config 파일 만들기
```
vim ~/.ssh/config
```

4. 접속 테스트
```sh
ssh -T git@gitlab.com
```
하면 

`Are you sure you want to continue connecting (yes/no)?`  
라는 메시지가 나오는데 yes하면 영구적으로 knownhost에 등록되었다고 나온다.

5. clone 테스트
private repository 정상적으로 clone됨






# Appendix
## Reference
> <https://mygumi.tistory.com/96>  
> ssh-agent 사용하는 이유 <https://devlog.jwgo.kr/2019/04/17/ssh-keygen-and-ssh-agent/>  
> <https://www.lainyzine.com/ko/article/creating-ssh-key-for-github/>  