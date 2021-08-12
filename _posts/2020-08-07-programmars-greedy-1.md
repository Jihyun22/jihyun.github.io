---
title: "[programmars] 프로그래머스 탐욕법(Greedy) kit - #1"
date: 2020-08-07
categories: programmars
toc: true
toc_sticky: true
toc_label: "목차"
tags : programmars algorithm python contest
header:
  teaser: "https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/teasers/programmers.png?raw=true"
---

<p>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=jihyun22.github.io/programmars/programmars-greedy-1/" alt="visitor"/>
</p>

<img src= "https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/programmars.png?raw=true">

Programmars(프로그래머스)의 코딩테스트 연습 탐욕법(Greedy) kit의 첫번째 문제, <체육복>의 문제와 풀이입니다.

- 프로그래머스 코딩테스트 연습 탐욕법 kit [바로가기](https://programmers.co.kr/learn/courses/30/parts/12244)
- 체육복 문제 [바로가기](https://programmers.co.kr/learn/courses/30/lessons/42862)



## 🎯 문제

점심시간에 도둑이 들어, 일부 학생이 체육복을 도난당했습니다. 다행히 여벌 체육복이 있는 학생이 이들에게 체육복을 빌려주려 합니다. 학생들의 번호는 체격 순으로 매겨져 있어, 바로 앞번호의 학생이나 바로 뒷번호의 학생에게만 체육복을 빌려줄 수 있습니다. 예를 들어, 4번 학생은 3번 학생이나 5번 학생에게만 체육복을 빌려줄 수 있습니다. 체육복이 없으면 수업을 들을 수 없기 때문에 체육복을 적절히 빌려 최대한 많은 학생이 체육수업을 들어야 합니다.

전체 학생의 수 n, 체육복을 도난당한 학생들의 번호가 담긴 배열 lost, 여벌의 체육복을 가져온 학생들의 번호가 담긴 배열 reserve가 매개변수로 주어질 때, 체육수업을 들을 수 있는 학생의 최댓값을 return 하도록 solution 함수를 작성해주세요.

### 제한사항

- 전체 학생의 수는 2명 이상 30명 이하입니다.
- 체육복을 도난당한 학생의 수는 1명 이상 n명 이하이고 중복되는 번호는 없습니다.
- 여벌의 체육복을 가져온 학생의 수는 1명 이상 n명 이하이고 중복되는 번호는 없습니다.
- 여벌 체육복이 있는 학생만 다른 학생에게 체육복을 빌려줄 수 있습니다.
- 여벌 체육복을 가져온 학생이 체육복을 도난당했을 수 있습니다. 이때 이 학생은 체육복을 하나만 도난당했다고 가정하며, 남은 체육복이 하나이기에 다른 학생에게는 체육복을 빌려줄 수 없습니다.

### 입출력 예

| n    | lost   | reserve   | return |
| ---- | ------ | --------- | ------ |
| 5    | [2, 4] | [1, 3, 5] | 5      |
| 5    | [2, 4] | [3]       | 4      |
| 3    | [3]    | [1]       | 2      |



### Test Case1

```python
1번 학생이 2번 학생에게 체육복을 빌려주고, 3번 학생이나 5번 학생이 4번 학생에게 체육복을 빌려주면 학생 5명이 체육수업을 들을 수 있습니다.
```

### Test Case2

```python
3번 학생이 2번 학생이나 4번 학생에게 체육복을 빌려주면 학생 4명이 체육수업을 들을 수 있습니다.
```



## 🔍 풀이

### 해설

기본적인 아이디어에서 출발하여 코드 길이를 압축하는 방법까지 풀이하겠습니다.

이 문제의 포인트는 "바로 앞 번호의 학생이나 바로 뒷번호의 학생"에게만 체육복을 빌려줄 수 있다는 것입니다. 따라서 선행 탐색 방법과 후행 탐색 방법 두가지를 고려해야 합니다.

풀이 방법은 학생의 체육복 수를 list로 두는 방법과 이미 주어진 lost와 reserve의 항목을 최대한 이용하는 방법 두가지로 나눌 수 있습니다.

### 기본 풀이

우선 학생의 체육복 수를 list로 두는 방법은

1. 학생 수 n 만큼의 index를 가지는 < 체육복 수 list >를 1로 초기화하고
2. lost한 학생의 번호에 < 체육복 수 list >를 -1,
3. reserve 한 학생의 번호에 < 체육복 수 list >를 +1 합니다.
4. 선행 탐색으로 뒷 번호의 학생이 바로 앞번호의 학생에게 체육복을 빌리는 경우를 고려,
5. 후행 탐색으로 앞 번호의 학생이 바로 뒷번호의 학생에게 체육복을 빌리는 경우를 고려하여 < 체육복 수 list >를 업데이트합니다.
6. 이후 < 체육복 수 list >의 item이 0 이상인 학생의 수를 count 하여 체육수업을 들을 수 있는 학생의 최댓값을 구할 수 있습니다.

### 응용 풀이

다음은 이미 주어진 lost와 reserve 항목을 최대한 이용하는 방법입니다.

1. lost에만 있는 학생 번호를  < lost list >, 
2. reserve에만 있는 학생 번호를  < reserve list >로 초기화합니다.
3.  < reserve list >에 있는 학생 번호의 -1, +1이 각각 lost에 존재하는지 확인하고
4. 존재하는 경우 해당 번호를 lost에서 삭제합니다.
5. 탐색이 완료된 후, 전체 학생 수 n에서 < lost list >의 item개수를 빼면 체육 수업을 들을 수 있는 학생의 최댓값을 구할 수 있습니다.



## 📝 Code

사용 언어는 <code><img height="25" src="https://github.com/Jihyun22/Jihyun22.github.io/blob/master/assets/images/python.png?raw=true">python3</code>입니다.

### 기본 풀이

```python
# python
# 2020-09-07
# [programmars] 프로그래머스 탐욕법(Greedy) kit - 1
# @Jihyun22

def solution(n, lost, reserve):
    # 초기화
    answer = 0
    student = [1 for i in range (n)]
    # lost 인 학생의 체육복 수를 -1, reserve인 학생의 체육복 수를 +1
    for i in range(len(student)):
        if i+1 in lost:
            student[i] -= 1
        if i+1 in reserve:
            student[i] += 1
    # 선행
    for i in range(len(student)-1):
        if student[i] < 1:
            if student[i+1] > 1:
                student[i] +=1
                student[i+1] -=1
    # 후행    
    for i in range(len(student)-1):
        if student[len(student)-i-1] < 1:
            if student[len(student)-i-2] > 1:
                student[len(student)-i-1] +=1
                student[len(student)-i-2] -=1
    # 수업을 들을 수 있는 학생 수
    for a in student:
        if a > 0: answer +=1
    # 출력
    return answer
```



### 응용 풀이

```python
# python
# 2020-09-07
# [programmars] 프로그래머스 탐욕법(Greedy) kit - 1
# @Jihyun22

def solution(n, lost, reserve):
    answer = 0
    # lost에만 있는 학생 번호
    lost_list = [l for l in lost if l not in reserve ]
    # reserve에만 있는 학생 번호
    reserve_list = [r for r in reserve if r not in lost ]
    # < reserve list > 탐색
    for r in reserve_list:
        # < lost list > 업데이트
        if r - 1 in lost_list:      lost_list.remove(r - 1)
        elif r + 1 in lost_list:    lost_list.remove(r + 1)
    # 체육 수업을 들을 수 있는 학생의 최댓값
    answer = n - len(lost_list)
    return answer
```



