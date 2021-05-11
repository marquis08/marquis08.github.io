---
date: 2021-05-11 15:29
title: "Aligning Equations in mathjax"
categories: Mathjax
tags: Mathjax
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---


# Align Equations in mathjax format
$$\begin{align}
\sqrt{37} & = \sqrt{\frac{73^2-1}{12^2}} \\
 & = \sqrt{\frac{73^2}{12^2}\cdot\frac{73^2-1}{73^2}} \\ 
 & = \sqrt{\frac{73^2}{12^2}}\sqrt{\frac{73^2-1}{73^2}} \\
 & = \frac{73}{12}\sqrt{1 - \frac{1}{73^2}} \\ 
 & \approx \frac{73}{12}\left(1 - \frac{1}{2\cdot73^2}\right)
\end{align}$$

```
$$\begin{align}
\sqrt{37} & = \sqrt{\frac{73^2-1}{12^2}} \\
 & = \sqrt{\frac{73^2}{12^2}\cdot\frac{73^2-1}{73^2}} \\ 
 & = \sqrt{\frac{73^2}{12^2}}\sqrt{\frac{73^2-1}{73^2}} \\
 & = \frac{73}{12}\sqrt{1 - \frac{1}{73^2}} \\ 
 & \approx \frac{73}{12}\left(1 - \frac{1}{2\cdot73^2}\right)
\end{align}$$
```

# Inline and block format in mathjax format
This is inline-command as $$ inline-format $$.  
This is block-format:  
\\[ block-format \\]

```
This is inline-command $$ inline-format $$.  
This is block-command:  
\\[ block-format \\]
```


reference:  
>mathjax align equal sign: <https://www.physicsoverflow.org/15329/mathjax-basic-tutorial-and-quick-reference>


