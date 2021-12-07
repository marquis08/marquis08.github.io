---
date: 2021-11-04 00:20
title: "OpenSlide Usages"
categories: Competition Pathology Segmentation
tags: Competition Pathology Segmentation
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

OpenSlide lib 사용방법

사용했던 메서들 중에서만 뽑아봤다.


```py
# OpenSlide_obj를 svs 파일로 읽어왔을 때의 obj라고 가정한다.
OpenSlide_obj = openslide.OpenSlide(svs_path)

OpenSlide_obj.level_dimensions
# A list of (width, height) tuples, one for each level of the slide. level_dimensions[k] are the dimensions of level k
# ex. ((300, 300), (200, 200), (100, 100))

OpenSlide_obj.properties
# Metadata about the slide, in the form of a Mapping from OpenSlide property name to property value. Property values are always strings. OpenSlide provides some standard properties, plus additional properties that vary by slide format.

OpenSlide_obj.read_region(location, level, size)
# Return an RGBA Image containing the contents of the specified region.
# Unlike in the C interface, the image data is not premultiplied.
# ---------------------------------------------------------------------
# Parameters
# location (tuple) – (x, y) tuple giving the top left pixel in the level 0 reference frame
# level (int) – the level number
# size (tuple) – (width, height) tuple giving the region size

OpenSlide_obj.get_thumbnail(size)
# get_thumbnail(size)
# Return an Image containing an RGB thumbnail of the slide.
# ---------------------------------------------------------
# Parameters
# size (tuple) – the maximum size of the thumbnail as a (width, height) tuple
```


# Appendix
## Reference
> <https://openslide.org/api/python/>