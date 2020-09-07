---
title: "[programmars] 프로그래머스 탐욕법(Greedy) kit - #2"
date: 2020-09-07
categories: programmars
toc: true
toc_sticky: true
toc_label: "목차"
tags : greedy codingtest programmars algorithm python
---

<p>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=jihyun22.github.io/programmars/programmars-greedy-2/" alt="visitor"/>
</p>

<img src= "https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/programmars.png?raw=true">

Programmars(프로그래머스)의 코딩테스트 연습 탐욕법(Greedy) kit의 두번째 문제, <큰수만들기>의 문제와 풀이입니다.

- 프로그래머스 코딩테스트 연습 탐욕법 kit [바로가기](https://programmers.co.kr/learn/courses/30/parts/12244)
- 체육복 문제 [바로가기](https://programmers.co.kr/learn/courses/30/lessons/42883?language=python3)



## 🎯 문제

###### 문제 설명

어떤 숫자에서 k개의 수를 제거했을 때 얻을 수 있는 가장 큰 숫자를 구하려 합니다.

예를 들어, 숫자 1924에서 수 두 개를 제거하면 [19, 12, 14, 92, 94, 24] 를 만들 수 있습니다. 이 중 가장 큰 숫자는 94 입니다.

문자열 형식으로 숫자 number와 제거할 수의 개수 k가 solution 함수의 매개변수로 주어집니다. number에서 k 개의 수를 제거했을 때 만들 수 있는 수 중 가장 큰 숫자를 문자열 형태로 return 하도록 solution 함수를 완성하세요.

##### 제한 조건

- number는 1자리 이상, 1,000,000자리 이하인 숫자입니다.
- k는 1 이상 `number의 자릿수` 미만인 자연수입니다.

##### 입출력 예

| number     | k    | return |
| ---------- | ---- | ------ |
| 1924       | 2    | 94     |
| 1231234    | 3    | 3234   |
| 4177252841 | 4    | 775841 |

[출처](http://hsin.hr/coci/archive/2011_2012/contest4_tasks.pdf)



## 🔍 풀이

### 해설

기본 풀이의 경우 테스트케이스 10번이 시간 초과로 통과되지 않아 틀린 답안입니다. 그러나 해당 풀이를 아래 응용 풀이 방법과 비교하며 '시간복잡도를 고려한 효율적인 코드를 짜는 방법'에 대해 고민해보면 좋을 것 같아 덧붙입니다.

#### 기본 풀이

주어진 숫자 리스트에서 선행 탐색하되, 가능한 자리수를 고려하여 범위 별 max 값을 구하는 방법입니다. 내장함수 max와 list의 슬라이싱이 사용되어 시간 복잡도는 약 O(n^2) 정도로 효율적인 코드는 아닙니다.

기본 아이디어는

1. return할 문자열의 자릿수는 전체 문자열 list의 길이 - k 이므로,
2. 0 ~ 자릿수만큼 i번 반복하며 max값을 탐색합니다.
3. max값을 탐색해야 할 list의 범위는 현재 max값이 위치한 index에서 k+i+1까지이며,
4. 해당 구간의 max값은 누적되어 answer에 저장됩니다.
5. 만약 k+1이 전체 문자열 list의 길이와 같다면,
6. 탐색하지 않은 남은 문자열 list을 answer에 덧붙여 answer을 구할 수 있습니다.

#### 응용 풀이

다음은 문제 그대로 k개의 요소를 삭제하며 answer을 구하는 풀이입니다. answer을 stack으로 사용하며 pop연산으로 삭제를 수행할 수 있습니다.

1. number 요소를 하나씩 answer의 마지막 요소(가장 나중에 삽입된 item)과 비교합니다.
2. 이때 answer에 하나 이상의 요소가 존재하며, k가 0보다 크고, answer[-1]보다 number 요소가 더 큰 경우,
3. k값에 -1연산을 취하고,
4. answer에 pop연산으로 마지막 요소를 삭제합니다.
5. 비교 연산 종료 후, number 요소를 answer요소에 추가합니다.
6. 일련의 연산을 반복한 이후, k가 0이 아닐 경우
7. answer에서 k개의 요소를 뒤에서부터 삭제하여 answer을 구할 수 있습니다.



## 📝 Code

사용 언어는 <code><img height="25" src="https://github.com/Jihyun22/Jihyun22.github.io/blob/master/assets/images/python.png?raw=true">python3</code>입니다.

#### 기본 풀이

```python
# python
# 2020-09-07
# [programmars] 프로그래머스 탐욕법(Greedy) kit - 2
# @Jihyun22

def solution(number, k):
    answer = []
    # 문자열 리스트를 정수 리스트로 변환
    num = list(number)
    index = 0
    # len(answer) = len(num) - k
    for i in range(len(num)-k):
        if k+i == len(num):
            return ''.join(answer)+''.join(num[index:])
        # index ~ k+i+1까지 max값 탐색
        tmp = num[index:k+i+1]
        x = max(tmp)
        index = tmp.index(x) + index + 1
        answer += str(x)
    return ''.join(answer)
```



#### 응용 풀이

```python
# python
# 2020-09-07
# [programmars] 프로그래머스 탐욕법(Greedy) kit - 2
# @Jihyun22

def solution(number, k):
    answer = []
    for num in number:
        # answer[-1] : answer의 마지막 요소
        while answer and answer[-1] < num and k > 0:
            k -= 1
            answer.pop()
        answer.append(num)
    if k != 0:
        # answer에서 k개의 요소를 뒤에서부터 삭제
        answer = answer[:-k]
    return ''.join(answer)
```



