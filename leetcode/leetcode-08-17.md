---
title: "[LeetCode] Distribute Candies to People"
toc: true
widgets:
  - type: toc
    position: right
  - type: recent_posts
    position: right
  - type: followit
    position: right
sidebar:
  right:
    sticky: true
date: 2020-08-17 15:00:00
cover: /img/cover/leetcode_cover.png
thumbnail: /img/thumbnail/leetcode_th.png
categories:
  - Algorithm
  - leetcode
tags:
  - leetcode
  - algorithm
  - python
---



LeetCoding Challenge의 8월 17일 **'Distribute Candies to People'** 문제 풀이입니다. 

- August LeetCoding Challenge [*week-3-august-15th-august-21st*](https://leetcode.com/explore/challenge/card/august-leetcoding-challenge/551/week-3-august-15th-august-21st/)

<!-- more -->



## **🎯 문제**

> We distribute some number of `candies`, to a row of **`n = num_people`** people in the following way:
>
> We then give 1 candy to the first person, 2 candies to the second person, and so on until we give `n` candies to the last person.
>
> Then, we go back to the start of the row, giving `n + 1` candies to the first person, `n + 2` candies to the second person, and so on until we give `2 * n` candies to the last person.
>
> This process repeats (with us giving one more candy each time, and moving to the start of the row after we reach the end) until we run out of candies. The last person will receive all of our remaining candies (not necessarily one more than the previous gift).
>
> Return an array (of length `num_people` and sum `candies`) that represents the final distribution of candies.
>
> **Constraints:**
>
> - 1 <= candies <= 10^9
> - 1 <= num_people <= 1000

캔디의 개수와 사람의 수가 주어지고, 사람에게 캔디를 분배하는 문제입니다. 사람에게 캔디를 1, 2, 3 ...개 순서로 분배하며, 마지막 사람까지 나눠준 다음 사탕이 남으면 다시 줄의 시작으로 돌아가 n+1, n+2 ... 순으로 재 분배합니다. 사탕이 다 떨어질 때 까지 이런 과정이 반복되며, 마지막 사람은 남은 사탕을 받습니다. 이때, 기존 사탕 분배 개수보다 반드시 한개 더 많이 받을 필요는 없습니다. 결과값은 사탕의 최종 분포를 리스트로 반환합니다.



### **Test Case1**

```python
Input: candies = 7, num_people = 4
Output: [1,2,3,1]
```

### **Test Case2**

```python
Input: candies = 10, num_people = 3
Output: [5,2,3]
```



## **🔍 풀이**

### 해설

1, 2, 3, 4 ... 배열의 누적 합 배열 1, 3, 6, 10 ... 을 고려하면 쉽게 접근할 수 있습니다.

사람의 수 만큼의 열을 가진 2차원 배열을 선언한 후, 사탕을 순서대로 분배하여 각 요소의 합을 연산해 결과값을 도출합니다.

| Test case 2 |  j=0   |  j=1  |  j=2  |
| :---------: | :----: | :---: | :---: |
|   **i=0**   | 1 (1)  | 2 (3) | 3 (6) |
|   **i=1**   | 4 (10) |       |       |
| **result**  | **5**  | **2** | **3** |



### **코드** 

사용 언어는 `python3`입니다.

```python
# python
# [LeetCode] week-3-august-17th
# Distribute Candies to People
# @Jihyun22

# 사탕 수와 사람 수가 각각 공백을 기준으로 입력되는 경우
c, p = list(map(int, input().split(" ")))
result = [0] * p

# candy_sum이 처음으로 10**9 이상일 때 i는 45
candy = 1
candy_sum = 0
tmp = [[0] * p for _ in range(45)]
for i in range(45):
    for j in range(p):
        if candy_sum + candy > c:
            tmp[i][j] = c - candy_sum
            candy = -1
            result[j] += tmp[i][j]
            break
        tmp[i][j] = candy
        candy_sum += candy
        candy += 1
        result[j] += tmp[i][j]
    # 이중 for 문 종료
    if candy < 0:
        break

print(result)
```



## **📝 Submit**

LeetCode 제출 코드입니다.

```python
class Solution:
    def distributeCandies(self, candies: int, num_people: int) -> List[int]:
        result = [0] * num_people
        candy = 1
        candy_sum = 0
        tmp = [[0] * num_people for _ in range(45)]
        for i in range(45):
            for j in range(num_people):
                if candy_sum + candy > candies:
                    tmp[i][j] = candies - candy_sum
                    candy = -1
                    result[j] += tmp[i][j]
                    break
                tmp[i][j] = candy
                candy_sum += candy
                candy += 1
                result[j] += tmp[i][j]
            if candy < 0:
                break
        return result
```

*LeetCode는 사용 언어 별 default 형식으로 작성해야 평가가 진행됩니다.* 



### **채점 결과**

![leetcode-08-18-Runtime](https://jihyun22.github.io/img/lc/0817.JPG)



채점 결과, 가장 보편적인 방법으로 접근한 것으로 판단됩니다. 상위 코드를 통해 run time을 줄일 수 있는 방법을 고민해봐야겠습니다.  아래에 run time = 20ms인 상위 코드 중 일부를 첨부합니다.

```  python
lo, hi = 0, candies
        K = 0
        while lo <= hi:
            k = (lo + hi)//2
            if k*(num_people*(num_people+1))//2 + (k*(k-1))//2 * num_people**2 <= candies:
                K = k
                lo = k + 1
            else:
                hi = k - 1
        result = [(i+1)*K+num_people*(K*(K-1))//2 for i in range(num_people)]
        candies -= sum(result)
        for i in range(num_people):
            add = min(candies, K * num_people + i + 1)
            result[i] += add
            candies -= add
            if candies == 0:
                break
        return result  
```



---



**관련 카테고리 포스트 더보기**

> [Algorithm 관련 포스트 더보기](https://jihyun22.github.io/categories/Algorithm/)
>
> [Leetcode 관련 포스트 더보기](https://jihyun22.github.io/categories/Algorithm/leetcode/)

