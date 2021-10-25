---
date: 2021-04-29
title: "Change default shell in linux"
categories: Linux
tags: Linux shell zsh bash
# 목차
toc: False  
toc_sticky: true 
toc_label : "Index"
---

Check current default shell
```
ps -p $$
# or
echo $0
# in my case
# /bin/bash
```

Get all shells available
```
cat /etc/shells
# in my case
# /bin/sh
# /bin/bash
# /bin/rbash
# /bin/dash
# /usr/bin/tmux
```

Swtich to different shell
```
# change to bash shell
bash
# or
chsh -s /bin/bash # chsh: change shell
```
to find full path of shell
```
type -a bash
# bash is /bin/bash
```

## change to Oh my zsh
```
# zsh 설치
sudo apt-get install zsh

# 설치경로 확인
which zsh
#=> /usr/bin/zsh

# 기본 sh 변경
chsh -s $(which zsh)
# or
chsh -s /usr/bin/zsh
```

### change theme
go to ~/.zshrc

```
# edit by vim
vim ~/.zshrc
```

change theme to whatever you want
ZSH_THEME="agnoster"

### Change shell
just type
bash or zsh on CLI

### Showing only user name


> <https://stackoverflow.com/questions/31848957/zsh-hide-computer-name-in-terminal>


# VSCODE remote connection font broken
server에서는 잘 보이는데  
vscode remote connection으로 할때는 폰트가 깨진다.  
windows 환경이라서 그런거 같다.  

windows 환경에서 powerline font 설치  
download 받은 fonts-master directory로 가서 참고한 블로그대로 설치하면 된다.  
설치후 원하는 powerline font 중 하나를

vscode file -> prefererences -> settings  
or  
ctrl + ,
를 입력해서 setting에 들어가서  
terminal font를 검색해서  
Terminal › Integrated: Font Family
여기세 설치된 원하는 font명을 넣어주면 적용되는 것을 볼 수 있다.


개인적으로 Space Mono for Powerline, Roboto Mono for Powerline, Cousine for Powerline 중에서   
Cousine이 제일 적절해 보인다.


>window 환경에서 powerline font 설치  <https://slmeng.medium.com/how-to-install-powerline-fonts-in-windows-b2eedecace58>


> find default shell and change   <https://www.cyberciti.biz/faq/how-to-change-shell-to-bash/>

> ubuntu zsh & oh-my-zsh install <https://tutorialpost.apptilus.com/code/posts/tools/using-zsh-oh-my-zsh/>
