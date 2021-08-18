---
date: 2021-05-06 17:20
title: "오늘의 삽질 - insert image in minimal mistakes"
categories: AllForNotThing
tags: github-pages minimal-mistakes
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

아무 생각 없이 이미지를 _site/assets/images 에 이미지를 새로 삽입하려고 하니  
자꾸 이미지가 삭제되는 것이였다...  
vscode remote connection이 문제인가 싶어  
server 가서 dir에다 넣어도 보고  
connection도 다시 해봤지만 계속 삭제 되길래  

가만히 보니까 잘못된 dir에 넣고 있던 것...  

_site는 gitignore의 주석에 따르면 jekyll 이 실행될때 생성되는 파일을 저장하는 저장소 같은 공간인 것 같다.  
여기에 이미지를 넣으려고 하니 계속 지워진 것...  

github page dir에서 assets/images에 바로 넣어야 되는데  
어쩐지 html img src에 _site가 없더라...  

dir 좀 잘 보고 넣자...  

🤢🤢🤢🤢🤢🤢🤢🤢

