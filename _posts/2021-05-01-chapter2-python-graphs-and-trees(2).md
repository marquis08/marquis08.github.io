---
date: 2021-05-02 03:00
title: "Chapter2 - The Basics - Graphs and Trees(2)"
categories: Python Algorithms
tags: Algorithms Graphs Trees Dict Set Hash Adjacency-List Adjacency-Set Traps sum extend float bunch-pattern
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

> **Book - Python Algorithms by Magnus Lie Hetland**

# Implementing Trees

## Binary Tree Class
```python
class Tree: 
    def __init__(self, left, right): 
        self.left = left 
        self.right = right 
```
```python
#You can use the Tree class like this: 
>>> t = Tree(Tree("a", "b"), Tree("c", "d")) 
>>> t.right.left 
'c'
```

## Multiway Tree Class
```python
class Tree: 
    def __init__(self, kids, next=None): 
        self.kids = self.val = kids 
        self.next = next
```
```python
>>> t = Tree(Tree("a", Tree("b", Tree("c", Tree("d"))))) 
>>> t.kids.next.next.val 
'c'
```

Multiway Tree<br>
<img src="/assets/images/multiway-tree.png" alt="Multiway Tree" >

# The Bunch Pattern
A flexible class that will allow you to specify arbitrary attributes in the constructor.
```python
class Bunch(dict): 
    def __init__(self, *args, **kwds): 
        super(Bunch, self).__init__(*args, **kwds) 
        self.__dict__ = self
```
This can create and set arbitrary attributes by supplying them as command-line arguments:
```python
>>> x = Bunch(name="Jayne Cobb", position="Public Relations") 
>>> x.name 
'Jayne Cobb'
```
By subclassing dict, you get lots of functionality for free, such as iterating over the keys/attributes or easily checking whether an attribute is present.  
Here’s an example: 
```python
>>> T = Bunch 
>>> t = T(left=T(left="a", right="b"), right=T(left="c")) 
>>> t.left {'right': 'b', 'left': 'a'} 
>>> t.left.right 
'b' 
>>> t['left']['right'] 
'b' 
>>> "left" in t.right # 'c'
True 
>>> "right" in t.right 
False
```

# Two Traps to be aware of.
## 1. Hidden Performance Traps: seems innocent but turining a linear operation into a quadratic one.
## 2. Floating Points Numbers

## Hidden Squares

### List VS. Set
```python
>>> from random import randrange 
>>> L = [randrange(10000) for i in range(1000)] 
>>> 42 in L 
False
>>> S = set(L) 
>>> 42 in S 
False
```
Checking memberships is *linear* for **lists** and *constant* for **sets**.  
If repeated, a list gives *quadratic* running time, whereas *linear* for sets.  
In short, it's important to pick the right built-in data structure for the specific job.

### Deque VS. Insert from the beginning in List
```python
>>> s = "" 
>>> for chunk in string_producer(): 
...     s += chunk
```
This works pretty well up to a certain size, but the optimizations break down requiring quadratic growth.  
The problem is that you need to create a new string for every += operation, copying the previous one.  
A better solution would be the following:  
```python
>>> chunks = [] 
>>> for chunk in string_producer():
...     chunks.append(chunk) 
... 
>>> s = ''.join(chunks)
```
Simplified version is:
```python
>>> s = ''.join(string_producer())
```

This is quadratic running time that manage to hide.
```python
>>> s = sum(string_producer(), '') 
Traceback (most recent call last): 
... 
TypeError: sum() can't sum strings [use ''.join(seq) instead]
```
Using list for an alternative solution above. (2d list)
```python
>>> lists = [[1, 2], [3, 4, 5], [6]] 
>>> sum(lists, []) 
[1, 2, 3, 4, 5, 6]
```
Looks elegant but really isn't.  
The sum function has to do one addition after another.  
That's why you're right back at the quadratic running time of the += example for strings.  

Here's a better way:
```python
>>> res = [] 
>>> for lst in lists: 
...    res.extend(lst)
```
Just try timing both versions. As long as lists is pretty short, there won’t be much difference, but it shouldn’t take long before the sum version is thoroughly beaten.

### Experiment of List extend VS. Sum (2d list):
```python
import time
def py_sum(lst):
    return sum(lst, [])

def py_list(lst):
    res = []
    for l in lst:
        res.extend(l)
    return res

n = 10**3
input_list = [[0]*n for _ in range(n)]

ctime = time.time()
py_sum(input_list)
elapsed = time.time()
print("Sum elapsed:  {:.10f}".format(elapsed-ctime))

ctime = time.time()
py_list(input_list)
elapsed = time.time()
print("List elapsed: {:.10f}".format(elapsed-ctime))

# Sum elapsed:  9.1245174408
# List elapsed: 0.0109813213
```
The result is pretty much remarkable.

## The Trouble with Floats
>In the second volume of The Art of Computer Programming, Knuth says, “Floating point computation is by nature inexact, and programmers can easily misuse it so that the computed answers consist almost entirely of 'noise'.”<br>  

0.1 represents inexact number.
```python
print("{:.20f}".format(0.1))
# 0.10000000000000000555
```
```python
>>> sum(0.1 for i in range(10)) == 1.0 
False
```

Check for approximate equality.
```python
>>> def almost_equal(x, y, places=7): 
...     return round(abs(x-y), places) == 0 
... 
>>> almost_equal(sum(0.1 for i in range(10)), 1.0) 
True

```
Decimal Module
```python
>>> from decimal import * 
>>> sum(Decimal("0.1") for i in range(10)) == Decimal("1.0") 
True
```

```python
>>> from math import sqrt 
>>> x = 8762348761.13 
>>> sqrt(x + 1) - sqrt(x) 
5.341455107554793e-06 
>>> 1.0/(sqrt(x + 1) + sqrt(x)) 
5.3414570026237696e-06
```
(with the latter being more accurate).

# Summary
## Asymptotic Notation
Asymptotic notation is used to describe the growth of a function.  
This allows us evaluate the salient features of the running time of an algorithm in the abstract.
## Graphs
Graphs are abstract mathematical objects consisting of a set of nodes, connected by edges.  
Edges can have direction and weight.  
In python programs, graphs can be represented as variations of adjacency lists and adjacency matrices, implemented with various combinations of *list*, *dict*, and *set*.
## Traps for Running Time
Built-in Python functions can give you a quadratic running time rather than a linear one.  
Profiling can uncover these traps.
## Traps for accuracy
Careless use of floating point numbers.

