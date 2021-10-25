---
date: 2021-05-01 20:25
title: "Chapter2 - The Basics - Graphs and Trees(1)"
categories: PythonAlgorithms
tags: Algorithms Graphs Trees Dict Set Hash Adjacent-List Numpy
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

> **Book - Python Algorithms by Magnus Lie Hetland**

# Dict and Set
## Hashing

Hashing involves computing seemingly random integer value from an arbitrary object.  
This value can be used as an index into an array.  
This mechanism is used in *dictionaries*, which are implemented using so-called *hash tables*.  
Sets are implemented using the same mechanism.
The important thing is that the hash value can be constructed in essentially *constant time*.  
It's constant with respect to the hash table size but *linear* as a function of the size of the object being hashed.  

In short, dict and set are having $$O(n)$$ time complextity

# Adjacency List and the Like

## Adjacency Sets
A Straightforward Adjacency Set Representation
```python
a, b, c, d, e, f, g, h = range(8) 
N = [ 
    {b, c, d, e, f},    # a 
    {c, e},             # b 
    {d},                # c 
    {e},                # d 
    {f},                # e 
    {c, g, h},          # f 
    {f, h},             # g
    {f, g}              # h 
]

```
N[v] is a set of v's neighbors.
```python
>>> b in N[a]  # Neighborhood membership 
True 
>>> len(N[f])  # Degree 
3
```

## Adjacency Lists
Overhead를 더 줄이는 방법중 하나는  
Adjacency sets 대신 adjacency lists를 사용하는 것이다.

> **Overhead**  <br>In computer science, overhead is any combination of excess or indirect computation time, memory, bandwidth, or other resources that are required to perform a specific task. It is a special case of engineering overhead. Overhead can be a deciding factor in software design, with regard to structure, error correction, and feature inclusion. Examples of computing overhead may be found in functional programming[citation needed], data transfer, and data structures.  <br>오버헤드(overhead)는 어떤 처리를 하기 위해 들어가는 간접적인 처리 시간 · 메모리 등을 말한다. <br><https://en.wikipedia.org/wiki/Overhead_(computing)> <br><https://ko.wikipedia.org/wiki/%EC%98%A4%EB%B2%84%ED%97%A4%EB%93%9C>

```python
a, b, c, d, e, f, g, h = range(8) 
N = [ 
    [b, c, d, e, f],    # a 
    [c, e],             # b 
    [d],                # c 
    [e],                # d 
    [f],                # e 
    [c, g, h],          # f 
    [f, h],             # g 
    [f, g]              # h 
]
```

## Adjacency Dicts with Edge Weights
```python
a, b, c, d, e, f, g, h = range(8) 
N = [ 
    {b:2, c:1, d:3, e:9, f:4},    # a 
    {c:4, e:3},                   # b 
    {d:8},                        # c 
    {e:7},                        # d 
    {f:5},                        # e 
    {c:2, g:2, h:2},              # f 
    {f:1, h:6},                   # g 
    {f:9, g:8}                    # h 
]
```
```python
>>> b in N[a]  # Neighborhood membership 
True 
>>> len(N[f])  # Degree 
3 
>>> N[a][b]    # Edge weight for (a, b)
2
```

A more flexible approach is to use a dict as main structure allowing us to use arbitrary, hashable, node labels.
```python
N = { 
    'a': set('bcdef'), 
    'b': set('ce'), 
    'c': set('d'), 
    'd': set('e'), 
    'e': set('f'), 
    'f': set('cgh'), 
    'g': set('fh'), 
    'h': set('fg') 
}
```
If you drop the set in above dict, it's adjacency strings.

## Adjacency Matrices
```python
a, b, c, d, e, f, g, h = range(8) #     a b c d e f g h 
N = [
    [0,1,1,1,1,1,0,0], # a 
    [0,0,1,0,1,0,0,0], # b 
    [0,0,0,1,0,0,0,0], # c 
    [0,0,0,0,1,0,0,0], # d 
    [0,0,0,0,0,1,0,0], # e 
    [0,0,1,0,0,0,1,1], # f 
    [0,0,0,0,0,1,0,1], # g 
    [0,0,0,0,0,1,1,0]
] # h
```
Instead of checking whether b is in N[a], you would check whether the matric cell N[a][b] is "True".
```python
>>> N[a][b]    # Neighborhood membership 
1 
>>> sum(N[f])  # Degree 
3
```

Advantages of Adjacency Matrices:  
- Not allowing self-loops, the diagonal is all False.
- For an undirected graph, the adjacency matrix will be symmetric.

Expending adjacency matrices to store weights, instead of "True", simply store weights.
For an edge (u,v), let N[u][v] be the weight w(u,v), instead of "True".
For practical reasons, we let nonexistent edges get an infinite weight, as "**inf = float('inf')**"
```python
# A Weight Matrix with Infinite Weight for Missing Edges
a, b, c, d, e, f, g, h = range(8) 
inf = float('inf') #       a    b    c    d    e    f    g    h 
W = [
    [  0,   2,   1,   3,   9,   4, inf, inf], # a 
    [inf,   0,   4, inf,   3, inf, inf, inf], # b 
    [inf, inf,   0,   8, inf, inf, inf, inf], # c 
    [inf, inf, inf,   0,   7, inf, inf, inf], # d 
    [inf, inf, inf, inf,   0,   5, inf, inf], # e 
    [inf, inf,   2, inf, inf,   0,   2,   2], # f 
    [inf, inf, inf, inf, inf,   1,   0,   6], # g 
    [inf, inf, inf, inf, inf,   9,   8,   0]
] # h
```
```python
>>> W[a][b] < inf   # Neighborhood membership 
True 
>>> W[c][e] < inf   # Neighborhood membership 
False 
>>> sum(1 for w in W[a] if w < inf) - 1  # Degree # exclude diagonal by -1
5
```

## Numpy Array for Adjacency Matrix
```python
N = [[0]*3 for i in range(3)] 
print("Python list: \n",N)
import numpy as np 
N = np.zeros([3,3])
print("Numpy array: \n",N)

# Python list: 
#  [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
# Numpy array: 
#  [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]
```
For a relatively sparse graph, Using a sparse matrix in *scipy.sparse* module is recommended.
