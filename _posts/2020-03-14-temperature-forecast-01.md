---
title: "[데이콘 온도추정 대회 리뷰] 1편-데이터셋 구조"
date: 2020-03-14
categories: Dacon ect
toc: true
toc_sticky: true
toc_label: "목차"
tags : 온도추정대회 DACON ML 머신러닝
---

데이콘에서 20년 3-4월 진행된 [AI프렌즈 시즌1 온도 추정 경진대회](https://dacon.io/competitions/official/235584/overview/)에 대해 리뷰하며, 대회에서 제공하는 데이터셋의 구조와 기본 아이디어에 대해 알아보겠습니다.



## 0. 배경

![](https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/2020-04-27-temperature-forecast/1.JPG?raw=true)

데이콘은 데이터 분석, 또는 ML에 대한 대회를 다루는 케글과 비슷한 국내 플랫폼입니다. 앞으로 데이콘의 지난 대회를 리뷰하며 ML에 대한 기본적인 개념을 다루고자 합니다.

가장 먼저 살펴볼 대회는 올해 2020년 3월부터 4월 중순까지 진행된 AI프렌즈 시즌1 온도 추정 경진대회입니다. 이 대회는 전*국에 걸쳐 시도별 기상 관측소가 있지만, 각 지역 내에서도 대상과 위치에 따라 온도 차이가 나는 점을 개선*하고자 기획되었습니다. *저가의 센서로 관심 대상의 온도를 단기간 측정하여 기상청의 관측 데이터와 상관관계 모델을 만들고 온도를 추정하여 서비스*하는 것이 최종 목표입니다.

대회에 대한 자세한 정보는 아래의 링크를 참조하세요.

<center> [AI프렌즈 시즌1 온도 추정 경진대회](https://dacon.io/competitions/official/235584/overview/)</center>

---



## 1. 데이터 구성

### 1.1 데이터 설명

데이콘에서는 각 대회의 데이터 구성 관련 설명 영상을 제공하고 있습니다. 이번 대회에서 사용하는 데이터셋이 어떻게 구성되어 있는지 자세한 설명은 아래의 영상을 참고하세요.

[![데이터설명](https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/2020-04-27-temperature-forecast/2.png?raw=true)](https://youtu.be/ukzaKsnKfXw)

본 대회에서 사용하는 데이터셋은 DACON [링크](https://dacon.io/competitions/official/235584/data/)에서 다운받을 수 있습니다.

---



### 1.2. train

train 데이터부터 살펴보겠습니다. `train.csv` 파일에는 총 33일 간 기상청 데이터와 온도 센서 데이터 값이 저장되어 있습니다.

```python
train = pd.read_csv('train.csv', index_col=False,)
train = train.drop(['id'], axis=1)
train.shape
```

> (4752, 59)



우선 기상청 데이터는 강수량, 기온, 기압 등을 포함한 `X00` ~ `X39`  총 40개의 피처로 주어집니다. 33일간 누락된 값은 없습니다.

온도 센서 데이터 값은 `Y00` ~ `Y18` 총 19개가 존재합니다. `train.shape` 의 열 값이 59인 이유는 40개의 기상청 데이터와 19개의 온도 센서 데이터 값으로 구성되어 있기 때문입니다.



이제 각 피처의 누락값 범위를 확인해보겠습니다.

```python
# 데이터 처음 5번째 줄 출력
train.head()
```

![](https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/2020-04-27-temperature-forecast/3.JPG?raw=true)

```python
# 데이터 끝 5번째 줄 출력
train.tail()
```

![](https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/2020-04-27-temperature-forecast/4.JPG?raw=true)

 `Y00` ~ `Y17` 까지의 측정값은 30일간의 측정값만 존재하고, `Y18` 값은 30일 이후의 3일 간의 데이터만 있습니다. 우리가 최종 예측해야 하는 값은 `Y18` 값입니다.

---



### 1.3. test

test데이터를 살펴보겠습니다. `train.csv`는 `train.csv` 기간 이후 80일 간의 기상청 데이터가 주어집니다. 

```python
test = pd.read_csv('test.csv', index_col=False,)
test = test.drop(['id'], axis=1)
test.shape
```

> (11520, 40)



누락값은 없습니다.

```python
# 데이터 처음 5번째 줄 출력
test.head()
```

![](https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/2020-04-27-temperature-forecast/5.JPG?raw=true)



표로 정리하면 다음과 같습니다.

|         구분          |             train - 30일              |              train - 3일              |              test - 80일               |
| :-------------------: | :-----------------------------------: | :-----------------------------------: | :------------------------------------: |
| `Y00` ~ `Y17`  (18개) |                *공개*                 | <span style="color:red">비공개</span> | <span style="color:red">비공개</span>  |
|         `Y18`         | <span style="color:red">비공개</span> |                *공개*                 | <span style="color:blue">목표값</span> |
| `X00` ~ `X39`  (40개) |                *공개*                 |                *공개*                 |                 *공개*                 |

이러한 데이터의 특징을 바탕으로, 2편에서는 모델 구성을 위한 기본 아이디어를 고민해보겠습니다.



---

<center> <BIG>"[데이콘 온도추정 대회] 2편-기본 아이디어" </BIG> </center>

---



---

<center> <BIG>[데이콘리뷰] 관련 목록 바로가기 </BIG> </center>

---

