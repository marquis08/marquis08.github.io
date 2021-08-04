---
date: 2021-08-04 12:53
title: "tmux installation and commands"
categories: tmux
tags: tmux
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

`remote ssh`와 `remote container`로 작업하다보니  

local 인터넷이 끊기는 경우에 remote server에 있는 training 과정을 실시간으로 못보게 되는 경우가 발생하기 때문에  

server에는 tmux가 설치되어서 사용하고 있으나 `docker container`에는 설치가 안되어 있기때문에  

tmux 안쓴지도 한참되어서 정리하면서 다시 상기시킬겸 tmux 설치 및 사용법에 대해 적어놓는다.  

나중에 docker image 만들때는 jupyter 랑 tmux를 설치하는 명령어도 추가하면 좋을 듯.  

- 설치
    ```sh
    sudo apt install tmux
    ```

- container 에서 설치
    ```sh
    apt install tmux
    ```

- 세션 시작  
    ```sh
    tmux
    ```

- 세션 종료 (from inside session)
    ```sh
    exit
    ```

- 세션 종료 (from outside session)
    ```sh
    tmux kill-session -t {session_name}
    ```

- tmux session list
    ```sh
    tmux ls
    ```

- session detach (session is running and out from session)
    ```
    ctrl + b then d
    ```

- session attach (accesee to specific session)
    ```sh
    tmux attach -t {session_name}
    ```




# Reference
> <https://tmuxcheatsheet.com/>
> <https://jjeongil.tistory.com/1361>