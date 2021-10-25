---
date: 2021-08-19 00:52
title: "StarGAN V2 - hpf(high-pass-filter)"
categories: DevCourse2 FashionGAN_Proj
tags: DevCourse2 FashionGAN_Proj
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# hpf

The purpose of introducing masks is to accurately preserve face landmarks (i.e., eyes, nose) of a source image during translation. The hyperparameter w_hpf controls the strength of the preservation. If you increase the value of w_hpf, then the **generator preserve the source landmarks more precisely**, but it would decrease the reflection of reference styles.  

You can set w_hpf higher than 1 for strong source preservation.

<https://github.com/clovaai/stargan-v2/issues/70>