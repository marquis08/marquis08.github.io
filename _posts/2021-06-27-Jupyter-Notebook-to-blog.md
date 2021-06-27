---
date: 2021-06-27 03:48
title: "Jupyter Notebook to GithubBlog"
categories: Jekyll Jupyter
tags: Jekyll Jupyter
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Failed so far.. with Jupyter Lab


# Notebook to HTML
```
jupyter nbconvert --to html --template classic --no-prompt --output hello_simple.html ./simple.ipynb
```
if not,  

In my case, I use Jupyter Lab. There is `file`--> `Export Notebook As...` and choose one of the options.  

# Jupyter Notebook 전용의 Jekyll Layout을 만들자.
그러면 Jupyter Notebook을 포스트 할 때 마다 Javascript, CSS를 넣는 것을 자동화 할 수 있다. 그리고 Jekyll Layout 안에 Jupyter Notebook의 셀 내용을 넣고, 새로운 포스트에서는 이 부분만 바꿔 넣는 방식으로 포스트를 빠르게 생산할 수 있게 된다.  

Version 1.2.4


포스트 입니다.

아래는 Jupyter Notebook 코드 입니다.

<iframe src="/assets/iframes/jupyter-notebooks/2021-06-27-simple.html/">Jupyter Notebook</iframe>

# Reference
> <https://seungwubaek.github.io/blog/tips/jupyter_to_html/#8-1-jupyter-notebook-iframe>  

