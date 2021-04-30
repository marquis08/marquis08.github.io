---
date: 2021-05-01 00:32
title: "Chapter2 - The Basics - Python List"
categories: Python Algorithms
tags: Algorithms, array, List
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

> **Book - Python Algorithms**

# Chapter 2 - The Basics
## Asymptotic Notation (점근 표기법)
It's about running times.

Example.  
append VS. insert
```python
import time
count = 10**5
start = time.time()
nums = []
for i in range(count):
    nums.append(i)
nums.reverse()
elapse = time.time()
print("append: ",elapse-start)

start = time.time()
nums = []
for i in range(count):
    nums.insert(0,i)
elapse = time.time()
print("insert: ",elapse-start)

##### output #####
# append:  0.024924755096435547
# insert:  4.281127691268921
```
Adding items to the end of a list scaled better with the list size than inserting them at the front.

## Python List
### Array VS. List  
Commonalities:  
- collections of items
- having an order

Differences:  
- Array has an index, not List.
    - Array occupies consecutive memory spaces.
    - This property makes array to find element much faster than List.
- Think of Linked List
    - Each items are spreaded in memory.
    - Each items can be accessed by its address of memory, not index.

<!-- Array:
![no-alignment](/assets/images/array.png) -->
<figure class="align-center">
    <figcaption>Array</figcaption>
    <img src="/assets/images/array.png" alt="">
  
</figure> 

### Then, why python "List" has an index?
Actually, it's a dynamic array, not the list mentioned above.
Python's list is implemented like array.
As we know, we use python's list as a stack.

In short, python's list is an "array" with high-leveled functioned applied.






> Array VS. List: <https://velog.io/@choonghee-lee/%EB%B2%88%EC%97%AD-Array-vs.-List-vs.-Python-List>
