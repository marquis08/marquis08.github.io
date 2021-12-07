---
date: 2021-11-29 14:55
title: "conda export & import by yaml"
categories: conda yaml
tags: conda yaml
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

`conda env export > environment.yaml`  or  

`conda env export --no-builds | grep -v “prefix” > environment.yaml`

`conda env create --file environment.yaml`

