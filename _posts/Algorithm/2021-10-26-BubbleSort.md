---
date: 2021-10-26 03:22
title: "Bubble Sort Implementation"
categories: Algorithm Sort
tags: Algorithm Sort
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

# Pseudo Code
1. adjacent elements 끼리 비교한다. 
2. 어떻게 sorted가 되었는지 판단하는가(종료조건)
    2-1. 첫 pass를 통과하면 마지막 원소는 무조건 가장 큰 원소가 됨
    2-2. 따라서 pass 할 때마다 뒤에서부터 정렬이 되는 알고리즘
    2-3. 그렇기 때문에, pass 후에 range는 뒤에서부터 1씩 줄어들게 하면 된다. (이게 포인트 ✔)
    2-4. 따라서 이중 반복문 수행후에는 종료가 된다. (원소 2개씩 비교하기 때문에)
    2-5. 안쪽 반복문은 index 1부터 시작해서 range end 지점이 1씩 줄어야 한다.
    2-6. 비교할 index를 프린트 해보면 이렇게 나와야 한다.
    (0, 1) (1, 2) (2, 3) (3, 4)
    (0, 1) (1, 2) (2, 3) 
    (0, 1) (1, 2) 
    (0, 1)

## 실패
- index 설정 실패
  - 이유: i,j 두개를 동시에 해야된다고 생각했다.
  - 해결: pointer하나만 사용하면 된다. 어차피 adjacent이기 때문에 하나를 기준으로 +1 해주면 되기 때문이다.
    - 사실 i의 용도는 j를 구하기 위해 필요했던 것 이다.
    - j를 -1 해주는 이유는 adjacent이기 때문에 j, j+1에서 인덱스가 넘어가 버리기 때문이다.
    - i는 위해서 얘기한 맨 마지막 원소들을 하나씩 줄이기 위해 사용

- 회고: arr로 넘어가기 전에 올바른 index 위치를 교환하는지 확인해 볼 것.


```py
myList = list(map(int, input().split()))
# print(myList)
for i in range(len(myList)):
    for j in range(len(myList)-i-1):
        print((j, j+1))
        if myList[j] > myList[j+1]: # need swap since the former is larger.
            myList[j], myList[j+1] = myList[j+1], myList[j]
print(myList)
```

# Appendix
## Reference
> <https://www.geeksforgeeks.org/bubble-sort/>