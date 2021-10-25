---
date: 2021-05-24 16:30
title: "Django HTML Extension with Issue & Solution"
categories: DevCourse2 Django HTML DevCourse2_Django
tags: DevCourse2 Django HTML DevCourse2_Django
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

# Django VScode Extension
While I work with django-html files, everytime I had to type curly brackets and percentage sign.  
I found most popular django html extension at VScode, but I keeps me bothering not allowing h2, h3 tags, and etc.  

Thus, I went to the publisher's git repository, and found the solution in Issue category.  

# Solution
Go to Settings by
```
ctrl+,
```  
- Type '**emmet.includeLanguages**'  
- Press **Add item**  
- Fill out Key and Value with:  
    - key: django-html  
    - value: html

Happy Coding 👍

ref:  
> django-html: <https://github.com/vscode-django/vscode-django/issues/50>