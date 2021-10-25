---
date: 2021-07-27 02:16
title: "Jekyll serve error: address already in use - bind(2) for 127.0.0.1:4000"
categories: Jekyll MinimalMistakes
tags: Jekyll MinimalMistakes
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Problem
After unexpectedly reload vscode for updating vscode,  
as I served jekyll from local host by `bundle exec jekyll serve`,  
`jekyll 3.9.1 | Error:  Address already in use - bind(2) for 127.0.0.1:4000` pops up.  


`jekyll 3.9.1 | Error:  Address already in use - bind(2) for 127.0.0.1:4000`:  
the port 4000 is already in use meaning that the process is not killed yet by closing the vscode terminal.  

# Solution
1. kill the process by finding process id (PID) 
   - find pid using port 4000 (lsof: list open files)
    ```sh
    $ lsof -wni tcp:4000
    ```
    - kill it
    ```sh
    kill -9 your_PID
    ```
2. Set another port as `jekyll serve --port 4001`.  

# Unix, Linux 에서 kill 명령어로 안전하게 프로세스 종료 시키는 방법
<https://www.lesstif.com/system-admin/unix-linux-kill-12943674.html>

`kill -9` 로 signal 을 보내면 개발자가 구현한 종료 함수가 호출되지 않고 즉시 프로세스가 종료되어 버리므로 데이타가 유실되거나 리소스가 제대로 안 닫히는 큰 문제가 발생할 수 있습니다.  

개인적으로 추천하는 방법은 먼저 `kill -TERM PID` 나 `kill -INT PID` 같이 종료를 의미하는 signal 을 여러 번 전송해 주는 것이며 제대로 된 프로그램은 보통 cleanup 코드를 수행하고 종료하게 됩니다.



