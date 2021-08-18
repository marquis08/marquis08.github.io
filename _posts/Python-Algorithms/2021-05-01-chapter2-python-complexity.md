---
date: 2021-05-01 04:01
title: "Chapter2 - The Basics - Python complexity"
categories: Python-Algorithms
tags: Algorithms timeit cprofile list-comprehension time-complexity
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

> **Book - Python Algorithms by Magnus Lie Hetland**

# Loops and Recursion
Loops and recursion are important for calculating complexity.

# List Comprehension Big-O
List Comprehension = $$O(n)$$, Linear Complexity

# Algorithim Engineering
Algorithm engineering: efficiently implementing algorithms reducing the hidden constants in that asymptotic complexity.

# Python Profiler and timeit
To find bottlenecks, use a profiler.  
*cProfile*
```python
import cProfile
cProfile.run('main()')
```


<br>

```python
import timeit

def test():
    return "-".join(str(n) for n in range(1000))

t1 = timeit.timeit('test()', setup='from __main__ import test', number = 10000)
print(t1)
```

from cli
```
python -m timeit -s"import mymodule as m" "m.myfunction()"
```

> timeit: <https://brownbears.tistory.com/456>

> Hetland, Magnus Lie. Python Algorithms (p. 20). Apress. Kindle Edition. 