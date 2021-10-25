---
date: 2021-10-26 03:29
title: "Merge Sort Implementation"
categories: Algorithm Sort
tags: Algorithm Sort
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

# Pseudo Code

- MergeSort(arr[], l,  r)  
  - If r > l  

1. Find the middle point to divide the array into two halves:  
        middle m = l+ (r-l)/2
2. Call mergeSort for first half:   
        Call mergeSort(arr, l, m)
3. Call mergeSort for second half:
        Call mergeSort(arr, m+1, r)
4. Merge the two halves sorted in step 2 and 3:
        Call merge(arr, l, m, r)

## 재귀호출 구현
```py
def mergeSort(arr):
    if len(arr) > 1:
  
        # Finding the mid of the array
        mid = len(arr)//2
          
        # Dividing the array elements
        # into 2 halves
        L = arr[:mid]
        R = arr[mid:]
  
        # Sorting the first half
        mergeSort(L)
        # Sorting the second half
        mergeSort(R)
  
        # Copy data to temp arrays L[] and R[]
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
  
        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
```

## In-Place (TODO)
- Merge Sort in-place 구현은 typical하지 않기 때문에 많은 형태가 있다.
- 아래는 그 중 하나의 구현방식인 투포인터를 사용한 방식이다.
```py
def merge(arr, start, mid, end):
    start2 = mid + 1
 
    # If the direct merge is already sorted
    if (arr[mid] <= arr[start2]):
        return
 
    # Two pointers to maintain start
    # of both arrays to merge
    while (start <= mid and start2 <= end):
 
        # If element 1 is in right place
        if (arr[start] <= arr[start2]):
            start += 1
        else:
            value = arr[start2]
            index = start2
 
            # Shift all the elements between element 1
            # element 2, right by 1.
            while (index != start):
                arr[index] = arr[index - 1]
                index -= 1
 
            arr[start] = value
 
            # Update all the pointers
            start += 1
            mid += 1
            start2 += 1
 
 
def mergeSort(arr, l, r):
    if (l < r):
 
        # Same as (l + r) / 2, but avoids overflow
        # for large l and r
        m = l + (r - l) // 2
 
        # Sort first and second halves
        mergeSort(arr, l, m)
        mergeSort(arr, m + 1, r)
 
        merge(arr, l, m, r)
```

## Iterative (TODO)
```py
def merge(left, right):
    if not len(left) or not len(right):
        return left or right
 
    result = []
    i, j = 0, 0
    while (len(result) < len(left) + len(right)):
        if left[i] < right[j]:
            result.append(left[i])
            i+= 1
        else:
            result.append(right[j])
            j+= 1
        if i == len(left) or j == len(right):
            result.extend(left[i:] or right[j:])
            break
 
    return result
 
def mergesort(list):
    if len(list) < 2:
        return list
 
    middle = int(len(list)/2)
    left = mergesort(list[:middle])
    right = mergesort(list[middle:])

    return merge(left, right)
```

# Appendix
## Reference
> In-Place Merge Sort <https://www.geeksforgeeks.org/in-place-merge-sort/>  
> Iterative Merge Sort <https://www.geeksforgeeks.org/iterative-merge-sort/>