---
date: 2021-10-26 03:22
title: "Quick Sort Implementation"
categories: Algorithm Sort
tags: Algorithm Sort
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

# Pseudo Code
1. 기준점(pivot)을 정한다. (4가지 방법 존재)
   - Always pick first element as pivot.  
   - Always pick last element as pivot (implemented below)  
   - Pick a random element as pivot.  
   - Pick median as pivot. 
2. 기준점 보다 작은 데이터는 왼쪽, 큰 데이터는 오른쪽
3. 종료조건: `list`의 길이가 2보다 작을 경우

## 리스트를 사용한 재귀호출 구현
- list를 사용해서 재귀호출을 하는 형태의 구현은 메모리 사용 측면에서 비효율적이다.

```py
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    lesser_arr, equal_arr, greater_arr = [], [], []
    for num in arr:
        if num < pivot:
            lesser_arr.append(num)
        elif num > pivot:
            greater_arr.append(num)
        else:
            equal_arr.append(num)
    return quick_sort(lesser_arr) + equal_arr + quick_sort(greater_arr)
```

## In-Place
```py
def partition(start, end, array):
    
    # Initializing pivot's index to start
    pivot_idx = start
      
    # This loop runs till start pointer crosses 
    # end pointer, and when it does we swap the
    # pivot with element on end pointer
    while start < end:          
        # Increment the start pointer till it finds an 
        # element greater than  pivot 
        while start < len(array) and array[start] <= array[pivot_idx]:
            start += 1
        # Decrement the end pointer till it finds an 
        # element less than pivot
        while array[end] > array[pivot_idx]:
            end -= 1

        # If start and end have not crossed each other, 
        # swap the numbers on start and end
        if start < end:
            array[start], array[end] = array[end], array[start]
      
    # Swap pivot element with element on end pointer.
    # This puts pivot on its correct sorted place.
    # Returning end pointer to divide the array into 2
    array[end], array[pivot_idx] = array[pivot_idx], array[end]
    return end
# The main function that implements QuickSort 
def quick_sort(start, end, array):
      
    # if start and end is not overlapped, proceed.
    if (start < end):
        # p is partitioning index, array[p] 
        # is at right place
        p = partition(start, end, array)
          
        # Sort elements before partition 
        # and after partition
        quick_sort(start, p-1, array)
        quick_sort(p+1, end, array)
```


# Appendix
## Reference
> <https://www.geeksforgeeks.org/quick-sort/> 
> <https://www.daleseo.com/sort-quick/> 
> <https://choiseokwon.tistory.com/233>