---
title: "[데이콘 온도추정 대회 리뷰] 2편-기본 아이디어"
date: 2020-03-21
categories: Dacon ect
toc: true
toc_sticky: true
toc_label: "목차"
tags : 온도추정대회 LSTM DACON ML 머신러닝 
---

데이콘에서 20년 3-4월 진행된 [AI프렌즈 시즌1 온도 추정 경진대회](https://dacon.io/competitions/official/235584/overview/)에 대해 리뷰하며, 1편에서 다룬 내용을 바탕으로 기본 아이디어와 LSTM 모델 적용 결과를 알아보겠습니다.



1편과 이어지는 자료입니다. 자세한 내용은 아래를 참고해주세요.

---

<center> <BIG>[데이콘 온도추정 대회] 1편-데이터셋 구조 </BIG> </center>

---





## 2. 모델 데모

### 2.1. 기본 아이디어

앞에서 살펴본 데이터셋의 특징은 크게 두가지입니다.

- 데이터의 크기가 **상당히** 작다.
  - 기상청 데이터의 피처는 40개, 온도 센서 측정값까지 더하면 60개 남짓입니다.
- 일자 별 **누락된** 데이터가 있다.
  - `test` 의 80일치 기상청 데이터를 예측해야 하는데, `Y18` 은 3일치의 데이터만 주어져 있습니다.

따라서 **적은 데이터셋으로 높은 정확도의 정보를 어떻게 이끌어 낼 것인가**가 중요한 요점입니다. 이러한 특징을 바탕으로 기본 아이디어를 설계해보겠습니다.



위 표에서 누락된 데이터가 있는 영역을 나누어 (가) ~ (자) 총 9개의 섹션으로 구분했습니다. 

|         구분          | train - 30일                               | train - 3일                                | test - 80일                                 |
| :-------------------: | :----------------------------------------- | :----------------------------------------- | :------------------------------------------ |
| `Y00` ~ `Y17`  (18개) | (가) *공개*                                | (라) <span style="color:red">비공개</span> | (사) <span style="color:red">비공개</span>  |
|         `Y18`         | (나) <span style="color:red">비공개</span> | (마) *공개*                                | (아) <span style="color:blue">목표값</span> |
| `X00` ~ `X39`  (40개) | (다) *공개*                                | (바) *공개*                                | (자) *공개*                                 |

1. X (다) + y (가) 모델1 학습**
   - 30일 간의 `X00` ~ `X39` 의 기상청 데이터 값으로 30일 간의 `Y00` ~ `Y17`  의 온도 센서 측정 값을 학습시켜 모델1을 만듭니다.
2. **모델1 : X (바) →  y (라) 예측**
   - 3일 간의 `X00` ~ `X39` 의 기상청 데이터 값으로 3일 간의 `Y00` ~ `Y17`  의 온도 센서 측정 값을 예측할 수 있습니다.
3. **X (바+라) + y (마) 모델2 학습**
   - 3일 간의 `X00` ~ `X39` 의 기상청 데이터 값과 3일 간의 `Y00` ~ `Y17`  의 온도 센서 측정 값으로 `Y18` 의 온도 센서 측정 값을 학습시켜 모델2을 만듭니다.
4. **모델1 : X (자) →  y (사) 예측**
   - 80일 간의 `X00` ~ `X39` 의 기상청 데이터 값으로 80일 간의 `Y00` ~ `Y17`  의 온도 센서 측정 값을 예측할 수 있습니다.
5. **모델2 : X (자+사) →  y (아) 예측**
   - 80일 간의 `X00` ~ `X39` 의 기상청 데이터 값과 80일 간의 `Y00` ~ `Y17`  의 온도 센서 측정 값으로  80일 간의 `Y18`  의 온도 센서 측정 값을 예측할 수 있습니다. 



작은 데이터 크기를 보정하기 위해 모델 2는 기상청 데이터 뿐 아니라 `Y00` ~ `Y17`  의 온도 센서 측정 값을 포함시켜 학습하고자 합니다. 따라서 목표값인 (아) 예측 성능은 **(사)의 예측 정확도에 좌우될 수 있다**는 점에 유의해야 합니다.

---



### 2.2. 모델 선정

기본 아이디어를 바탕으로 모델 데모를 작성하기 위해 머신러닝 모델을 선정하겠습니다.

예측 값이 2차원인 모델 1은 3차원 분석이 가능한 `LSTM`이 적절하다고 판단됩니다.

모델2는 예측 값이 `Y18` 하나로 1차원이기에 다양한 모델이 선호될 수 있습니다. 우선 데모를 위해 간단하게 모델 1과 모델 2 모두 `LSTM` 를 적용해보겠습니다.

`LSTM`에 대한 자세한 내용은 아래 자료를 참고하세요

- [[모델] LSTM 에 대해서](tmp)

---



### 2.3 모델 학습

 전체적인 코드는 링크를 참조해주세요. 로컬에서 `Jupyter` 로 작업하였으며 workplace 내 `train.csv`, `test.csv` 를 위치시켰습니다.



#### 2.3.1 데이터 로드

우선 데이터를 로드하겠습니다. 모델1을 만들기 위한 데이터 구성입니다. 

```python
import pandas as pd
import numpy as np
from tqdm import tqdm

# 트레인셋
train = pd.read_csv('train.csv', index_col=False)
X_train = train.loc[:,'X00':'X39']
y_train = train.loc[:,'Y00':'Y17']

# 테스트 셋
test = pd.read_csv('test.csv', index_col=False)
test=test.drop(['id'], axis=1)

```

기상청 데이터값을 `X_train` 으로,   `Y00` ~ `Y17`  온도 센서 측정 값을 `y_train` 으로 로드했습니다. 



```python
# nan 값 제거
y = y_train.dropna()

# 트레인 셋 범주 조정
X = X_train.loc[:y.shape[0]-1,:]
```

이후, `y_train`의 결측치를 제거하여 30일 간의  `Y00` ~ `Y17`  온도 센서 측정 값을 `y` 로 저장했습니다. 모델 구축을 위해 33일 간의 기상청 데이터로 구성된  `X_train` 도 `y` 와 **같은 수의 row**를 가진 `X`로  조정합니다.

이제 데이터 프레임 `X` 와 `y` 는 각각 **30일 간**의 기상청 데이터와 온도 센서 측정 값으로 구성되어 있습니다. 

- `X.shape`  : *(4320, 40)*
- `y.shape`  : *(4320, 18)*



#### 2.3.2 모델 1 학습

다음은 `LSTM`모델을 세팅하겠습니다. 모든 파라미터는 `default` 값입니다.

```python
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping

K.clear_session()
model_1 = Sequential() # Sequeatial Model
model_1.add(LSTM(20, input_shape=(X.shape[1], 1))) # 트레인값
model_1.add(Dense(y.shape[1])) # 출력값
model_1.compile(loss='mean_squared_error', optimizer='adam')

model_1.summary()
```

![](https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/2020-04-27-temperature-forecast/6.JPG?raw=true)

`LSTM` 모델에 사용되는 학습 데이터와 출력값의 크기를 세팅하였습니다. 



이제 세팅한 학습 데이터의 크기에 맞게 `X` 를 조정하겠습니다.

```
X = X.values
X = X.reshape(X.shape[0], X.shape[1], 1)
```

`pandas`   `dataframe` 을 `LSTM`의 학습 데이터로 사용하기 위해서는 `.values` 로  `dataframe` 을 풀어준 후, `.reshape` 메소드로 (x, y, z) 값을 설정하면 됩니다.  



모델 학습을 진행하겠습니다. 일정 이상 loss값에 도달하면 `early stop` 되며 다른 파라미터는 기본값 세팅으로 사용했습니다.

```python
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model_1.fit(X, y, epochs=10000,
          batch_size=30, verbose=1, callbacks=[early_stop])
```

최종 loss값 <tt>5.869</tt>으로 학습을 마무리했습니다.



모델1은 기본 아이디어에 따라 이후 사용됩니다. 재사용의 편의를 위해 모델을 저장하고 적용 함수를 만들었습니다.

```python
#모델 저장
import joblib
joblib.dump(model_1, 'model_1.pkl')

# 모델 적용 함수
def model_fit_1(data, model_name, file_name):
    model = joblib.load(str(model_name)) 
    data_r = data.values.reshape(data.shape[0],data.shape[1],1 )
    pred_out=model.predict(data_r)
    df = pd.DataFrame(pred_out)
    df.to_csv(str(file_name), index = False, header = False)
    return df
```



이 모델로 지난 3일간의 `y` 값 (라)을 연산했습니다.

```python
#지난 3일간의 데이터셋
X_test = X_train.loc[y.shape[0]:,:]

#지난3일간의 y값 연산
pred_out = model_fit_1(X_test, 'model_1.pkl', 'pred_out_1.csv')
```



#### 2.3.3. 모델 2 학습

이제  *X (바+라) + y (마) 모델2 학습* 단계입니다. 

```python
X=pd.concat([X_test.reset_index(drop=True), pred_out.reset_index(drop=True)], axis=1)

y = train['Y18']
y = y.dropna()
```

우선 3일간의 `X` 값과 모델 1에서 연산한 (라)값을 합하여 `X` 를 구성하였습니다. `y` 는 3일 간의 `Y18` 온도 센서 측정 값입니다.



`LSTM` 모델을 세팅합니다.

```python
K.clear_session()
model_2 = Sequential() # Sequeatial Model
model_2.add(LSTM(20, input_shape=(X.shape[1], 1)))
model_2.add(Dense(1)) # output = 1
model_2.compile(loss='mean_squared_error', optimizer='adam')

model_2.summary()
```

![](https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/2020-04-27-temperature-forecast/7.JPG?raw=true)

모델 1과 동일하나, `y18` 의 크기를 고려하여  `output` 데이터의 크기를 1로 조정했습니다.



모델 1과 동일하게 세팅한 학습 데이터의 크기에 맞게 `X` 를 조정하겠습니다.

```
X = X.values
X = X.reshape(X.shape[0], 58, 1)
```

이때,  `X` 의 y값은 40개의 기상청 데이터와 18개의 온도 측정 센서를 더한 58로 설정합니다. 



모델 학습을 진행하겠습니다. 

```python
early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model_2.fit(X, y, epochs=10000,
          batch_size=30, verbose=1, callbacks=[early_stop])
```

최종 loss값 <tt>6.6453</tt>으로 학습을 마무리했습니다.



모델2 역시 기본 아이디어에 따라 이후 사용됩니다. 재사용의 편의를 위해 모델을 저장하고 적용 함수를 만들었습니다.

```python
joblib.dump(model_2, 'model_2.pkl')

# 모델2 적용 함수
def model_fit_2(data1, data2, model_name, file_name):
    data=pd.concat([data1.reset_index(drop=True), data2.reset_index(drop=True)], axis=1)
    model = joblib.load(str(model_name))
    data_r = data.values.reshape(data.shape[0],data.shape[1],1 )
    pred_out=model.predict(data_r)
    df = pd.DataFrame({'id':range(144*33, 144*113),
              'Y18':pred_out.reshape(1,-1)[0]})
    df.to_csv(file_name, index = False)
    return df
```

---



## 3. 평가

본 대회의 평가지표는 MSE입니다. MSE관련 내용은 다른 포스트에서 소개하겠습니다.



만든 모델을 평가하겠습니다. 80일 간의 기상청 데이터 값인 `test` 값으로 제출 파일을 만들었습니다.

```python
# 모델1 : X (자) →  y (사) 예측
pred_out_1 = model_fit_1(test, 'model_1.pkl', 'pred_out_1.csv')
# 모델2 : X (자+사) →  y (아) 예측
pred_out_fin = model_fit_2(test, pred_out_1, 'model_2.pkl', 'pred_out_fin.csv')
```

제출 결과, 8.9381423723점으로 약 9점에 가까운 점수를 받았습니다. 

데이터 전처리 없이 기본 LSTM 모델로 학습을 진행한 데모이기에, 9점이 좋은 모델의 지표가 될 수 있을거라 생각됩니다.



다음 단계에서는 데이터 전처리에 대해 알아보겠습니다.



---

<center> <BIG> [데이콘 온도추정 대회] 3편-탐색적 데이터 분석(EDA) </BIG> </center>

---



---

<center> <BIG>[데이콘리뷰] 관련 목록 바로가기 </BIG> </center>

---























