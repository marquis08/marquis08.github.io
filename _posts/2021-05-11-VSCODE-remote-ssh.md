---
date: 2021-05-11 20:20
title: "Remote-ssh access in VSCODE"
categories: remote-ssh vscode
tags: remote-ssh vscode ssh port-forwarding
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---


# Remote-ssh VScode
1. install VSCODE
2. press F1
3. Find Remote-SSH: Connect current window to Host (or Remote-SSH: Connect to Host)
4. 만약 최초 config 만들어야 되면  
    4-1. Configure SSH Hosts... 선택
5. 제일 위에 있는 config 파일 선택 (Users/..)
6. 연결 정보 입력    
    ex.
    ```jsx
    Host laptop1
        HostName 111.111.111.1
        User username
        Port 1234

    Host laptop2
        HostName 111.111.111.1
        User username
        Port 1234

    이런식으로 여러개의 host 목록 등록 가능
    ```
## HostName
HostName 같은 경우에는 공유기에서 ip를 확인해야 하는데,  
kt의 경우는 http://172.30.1.254 를 주소창에 입력하면 공유기 설정 로그인 화면에 접속이 가능하다.  
여기서 시스템 정보 - 인터넷 연결정보 - ip주소를 HostName에 입력하면 된다. (외부ip)

## Port
그 다음으로 포트포워딩을 설정해줘야 하는데  
KT 기준으로는 장치설정 - 트래픽 관리 메뉴에 들어가서  
설정을 해주면 되는데  
여기서 내부 ip가 필요하다  

내부 ip는 윈도우 환경이라면  
cmd에 들어가서  
ipconfig 명령어를 입력하면  
IPv4 주소가 나온다.

이걸 포트포워딩 설정창에 추가해 주면 된다.  

반드시 채워야 하는 부분:  
외부포트, 내부 ip 주소, 내부 포트, 프로토콜, 설명(option)  

여기서 외부포트를 config의 Port에 넣어주면 된다.  

## UserName
linux의 경우 username을 입력하면된다.  
윈도우의 경우는 안해봐서 모르겠다.  



# Port Forwarding (포트 포워딩)
출처는 아래에  

## 외부 ip
예를들어 외부 IP주소가 222.112.33.188 이라고 합시다.
이 IP주소는 다른집과 중복되지 않는 IP주소입니다.
집이 아닌 다른곳에서 저의컴퓨터로 접근을하려면

222.112.33.188이라는 IP주소로 접근을 해야합니다.

하지만 제가지금 사용하는 웹서버를 이용하기위해서는 공유기 IP인
172.30.1.43:8080 을 이용해야 합니다.

그렇다면 외부에서 222.112.33.188:8080 으로 접근을 하게된다면
172.30.1.43:8080으로 바로 연결이 될까요?

아닙니다.
222.112.33.188이라는 IP주소 안의 공유기IP주소가 여러개가 사용되고 있고, 휴대폰으로 와이파이를 연결하면
172.30.1.39

노트북으로 연결을하면
172.30.1.43

다른 노트북으로 연결을하면
172.30.1.41
와 같이 주소가 여러개로 나뉘게됩니다.

## 포트포워딩으로 접속하기
이중 우리는 172.30.1.43의 공유기 IP의 8080이라는 포트로 이동을 해야하는데 이것을 구분하는 방법을 정하는 것을 포트포워딩이라고 합니다.

외부IP에 포트번호를 어떠한번호를 입력하면,
사용자가 정해둔 공유기IP의 원하는 포트번호로 이동하게 되는것이죠.


reference:  
> <https://evols-atirev.tistory.com/28>  
> <https://mclearninglab.tistory.com/171>  
> port-forwarding: <https://m.blog.naver.com/PostView.nhn?blogId=pk3152&logNo=221380441554&proxyReferer=https:%2F%2Fwww.google.com%2F>  
