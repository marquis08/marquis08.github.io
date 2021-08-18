---
date: 2021-05-05 02:39
title: "Git Pull Request (PR) - Project Contribution"
categories: DevCourse2
tags: Git Github PR DevCourse2
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

# Git Pull Request
다 적고보니 너무 자세하게 적은 것 같기도 하다.  

## Fork or Download zip

1. original repo를 fork하거나 혹은 download 받아서 remote를 다시 연결 시켜줘야한다.  
1-1. download보다 fork 후 clone하면 remote가 자동으로 연결되기 때문에 fork - clone을 추천한다.  
1-2. [fork version] original repo의 fork 버튼을 클릭하면 나의 repo로 추가된다.  
1-3. [fork version] 추가된 repo에서 code 버튼을 누르면 ~/~.git 주소가 나온다.
1-4. [fork version] 이 주소 복사 후 terminal CLI에서 원하는 directory로 이동한다.  
1-5. [fork version] 해당 dir에서 terminal CLI에 "git clone 주소"를 입력하면 파일들이 현재 dir 아래에 폴더가 만들어지면서 파일들이 생성된다.  

## Branch check and work

2. Branch가 main한개만 있을 수도 있고 다른 branch가 있을 수도 있다. 원하는 branch로 이동.  
2-1. 원하는 branch 이동은 "git checkout 브랜치이름"  
2-2. 브랜치 리스트를 보고싶다면: "git branch --list" 실행  
2-3. 해당 브랜치로 checkout  
2-4. 기존 branch를 intact하게 놔두고 새로운 branch를 파고 싶다면(추천) "git checkout -b 브랜치이름" 실행하면 자동으로 생성하고 해당 브랜치로 이동시켜준다.  
2-5. 해당 branch 에서 "ll" 입력하면 현재 branch에 있는 file listing  
2-6. 파일 수정 작업~~  


## Add and Commit

3. "git status"로 어떤 파일들을 작업했는지 확인 가능하다.  
3-1. git status 결과화면을 보면  
    - On branch your-branch-name
    - Untracked files: ~~ (use "git add <file> .." ~)
    - 만약 git add한 파일이 있다면 changes to be commited: 파일 리스트
    - unstage하고 싶다면 "git reset HEAD <file> .."
    - 등등의 결과를 보여준다.  
3-2. "git add file" 명령어로 commit 하기전에 staged를 하는게 좋다.(추천) 만약에 여러 파일이 동일한 수정내용이라면 "git add --all"로 한꺼번에 staging도 가능하다.  
3-3. "git status"로 다시 현재 staged된것과 안된것을 볼 수 있다.  
3-4. staged된 파일들은 green highlighted syntax가 적용된 것을 알 수 있다.  
3-5. staged된 파일중에서 commit하고 싶은 파일을 "git commit file -m 'Write a message of this commit'" -m args를 통해서 메시지와 함께 commit 한다.  


## Push to remote repo

4. 원격 저장소(remote repo)에 local에서 작업한 내용을 push하고자 한다.  
4-1. "git push origin main" main대신 branch이름을 쓰고 실행하면 된다.  
4-2. 에러가 없이 잘 실행되었는지 확인  
4-3. 만약에 remote repo에 없는 branch를 새로 push하고자 하면 error가 발생하면서 upstream을 해주라는 메시지가 뜰 것이다.  
4-4. local에만 존재하는 branch를 push 하려면 "git push -u origin branchname" 실행하면 정상적으로 push가 된다. 이는 첫번째 실행시만 하면 된다.  

## Pull Request (PR)
5. 대망의 PR이다. 나의 forked 된 origin repo의 main 브랜치에서 작업한 내용을 upstream에 합치고 싶다.  
5-1. 나의 forked 된 repo의 main 브랜치를 선택하면 **compare & pull request** 버튼이 보일 것이다. (만약 다른 브랜치에서 작업했다면 다른 branch로 가서 pull request를 하면 된다.)  
5-2. 안보인다면 new pull request 버튼을 누르자.  
5-3. pull request 버튼을 누르면 upstream 브랜치와 나의 브랜치를 선택할 수 있는 화면이 나온다.  
5-4. 브랜치를 정확히 선택한 후 create pull request를 누르면 내가 commit한 것들과 message를 작성할 수 있는 화면이 나온다. 이는 repo의 원래 주인에게 이러저러한 변경사항을 업데이트 했으니 봐주세요 하는 것과 동일하다. 내가 PR을 했다고 하더라고 기존 repo 반영되는 것은 아니다. 기존 repo의 주인이 승인을 해야 confirm이 되면서 반영이 된다. 여기서 confirm or reject가 되는 부분이다.  

## PR이 되면 contributor가 되나보다.



### Upstream 이란 ?

다른 사람의 GitHub의 저장소를 Fork한 경우 내 GitHub가 origin이 됩니다. 여러분이 처음 fork를 시도한 저장소를 upstream이라고 부릅니다. origin와 upstream 모두 remote 저장소입니다. 보통 origin과 구분하기 위해서 upstream 이라는 명칭을 주로 사용합니다.  
요약: forked된 나의 저장소가 origin이 되었고, 기존 original 저장소는 upstream이 된 것이다.  

> upstream: <https://wikidocs.net/74836>

> 간략하게 요약한 블로그: <https://chanhuiseok.github.io/posts/git-3/>