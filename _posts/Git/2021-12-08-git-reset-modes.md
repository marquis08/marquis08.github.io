---
date: 2021-12-08 18:16
title: "Git Reset Modes: soft, mixed, hard"
categories: Git
tags: Git submodule
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

commit을 reset해서 이전 commit 지점으로 돌아갈 필요가 있는 경우가 존재한다.

주로 vscode를 사용해서 진행하면서 git graph를 사용하는데 reset 하려고 보니 3가지 옵션을 선택하라고 나온다.  

각 mode가 어떤 방식으로 진행되는지 알아보자.  

가장 직관적으로 이미지로 보면  
![git-reset-mode](https://www.cloudsavvyit.com/p/uploads/2021/07/f5026f58.png?trim=1,1&bg-color=000&pad=1,1)

official docs를 보자.  
- `--soft`
  - Does not touch the index file or the working tree at all (but resets the head to `<commit>`, just like all modes do). This leaves all your changed files "Changes to be committed", as `git status` would put it.
- `--mixed`
  - Resets the index but not the working tree (i.e., the changed files are preserved but not marked for commit) and reports what has not been updated. This is the default action.
  - If `-N` is specified, removed paths are marked as intent-to-add (see [git-add[1]](https://git-scm.com/docs/git-add)).
- `--hard`
  - Resets the index and working tree. Any changes to tracked files in the working tree since `<commit>` are discarded. Any untracked files or directories in the way of writing any tracked files are simply deleted.

- `soft mode`: staged 상태로 돌아가게 된다.
- `mixed mode`: staged 되기 전 상태로 돌아가게 된다.
- `hard mode`: changes 들은 다 삭제되고 untracked files들도 삭제 된다.


# Appendix
> git reset: <https://www.cloudsavvyit.com/12962/how-does-git-reset-actually-work-soft-hard-and-mixed-resets-explained/>
> 
> git reset official: <https://git-scm.com/docs/git-reset>

