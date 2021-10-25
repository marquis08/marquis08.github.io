---
date: 2021-06-01 19:32
title: "Minimal Mistakes Utility - Image Size Modification"
categories: SCSS MinimalMistakes
tags: SCSS MinimalMistakes
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

# Minimal-Mistakes Style Sheet Structure
![scss](/assets/images/scss.jpg){: .align-center .img-50}  
Got a hint from:  
<https://github.com/mmistakes/minimal-mistakes/issues/1583>  
```
![](/images/image.jpg){: .align-right .width-half}
```
Go to **_utilities.scss**  
Add  
```css

.img-20 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 20%;
  height: 20%;
}

.img-30 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 25%;
  height: 25%;
}

...

.img-70 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 70%;
  height: 70%;
}

.img-80 {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 80%;
  height: 80%;
}
```


```
![GGU](/assets/images/GGU.jpg){: .align-center .img-20}  
```  
![GGU](/assets/images/GGU.jpg){: .align-center .img-20}  

```
![GGU](/assets/images/GGU.jpg){: .align-center .img-50}  
```  
![GGU](/assets/images/GGU.jpg){: .align-center .img-50}  

```
![GGU](/assets/images/GGU.jpg){: .align-center .img-70}  
```  
![GGU](/assets/images/GGU.jpg){: .align-center .img-70}  


# References
> Minimal-Mistakes: <https://mmistakes.github.io/minimal-mistakes/docs/stylesheets/>  
