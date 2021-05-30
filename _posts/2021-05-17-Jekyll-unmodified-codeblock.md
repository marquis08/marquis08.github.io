---
date: 2021-05-17 20:20
title: "Jekyll unmodifided code block"
categories: Jekyll markdown
tags: Jekyll markdown
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

While writing django code snippet, I found that it hides double curly brackets( {% raw %}{{}}{% endraw %} ).  
Since Jekyll uses the Liquid templating language to process templates, Liquid's template language uses double curly brackets, also.  

To show, use raw tag:  
```
{% raw %} {{ }} {% endraw %}
```  

Done!


ref:  
> stackoverflow: <https://stackoverflow.com/questions/24102498/escaping-double-curly-braces-inside-a-markdown-code-block-in-jekyll>