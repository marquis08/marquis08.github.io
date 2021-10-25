---
date: 2021-09-29 14:30
title: "python concurrent futures"
categories: Python
tags: Python
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

```py
import time
data = []
start = time.time()
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

with ThreadPoolExecutor(max_workers=60) as executor:
    future_to_json = {executor.submit(open_json, j): j for j in tr_json_list}
    for future in concurrent.futures.as_completed(future_to_json):
        j = future_to_json[future]
        try:
            data.append(future.result())
            
        except Exception as exc:
            print('%r generated an exception: %s' % (j, exc))
#         else:
#             print('%r page is %d bytes' % (j, len(data)))
elapsed = time.time() - start
print("Elapsed {}".format(elapsed))
```

<https://docs.python.org/3.7/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor>  
<https://medium.com/humanscape-tech/%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9D%98-future-%ED%81%B4%EB%9E%98%EC%8A%A4-8b6bc15bd6af>