---
date: 2022-02-11 11:00
title: "Git bash & conda in Windows"
categories: conda git bash
tags: conda git bash
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

1. Go to conda.sh location such as
   1. `C:\Users\{your_username}\anaconda3\etc\profile.d`
2. Open git bash here by right click as 
   1. `Git Bash Here`
3. Enter command
   1. `echo ". ${PWD}/conda.sh" >> ~/.bashrc`
4. Close Git Bash & Re-Open
   1. command `conda`




# Appendix
## Reference
> How to add Conda to Git Bash (Windows): <https://fmorenovr.medium.com/how-to-add-conda-to-git-bash-windows-21f5e5987f3d> 
>  
> 