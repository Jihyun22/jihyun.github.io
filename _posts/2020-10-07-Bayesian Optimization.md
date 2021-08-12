---
title: "[AUTO ML] 베이지안 최적화 (Bayesian Optimization)"
date: 2020-10-07
categories: automl
toc: true
toc_sticky: true
toc_label: "목차"
tags : data datastudy dl automl
related: true
header:
  teaser: "https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/teasers/Bayesian%20Optimization.png?raw=true"
---



<img src= "https://github.com/Jihyun22/Jihyun22.github.io/blob/master/_posts/images/teasers/Bayesian%20Optimization.png?raw=true">

이번 포스트에서는 Auto ML 방법 중 하이퍼 파라미터 튜닝에 대해 다루며,  Bayesian Optimization(베이지안 최적화) 방법을 소개하겠습니다.

- 참고자료 [링크](https://research.sualab.com/introduction/practice/2019/02/19/bayesian-optimization-overview-1.html) (본 포스트는 링크의 자료를 상당 부분 인용, 참고하여 작성되었습니다)

<center><p>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=jihyun22.github.io/automl/Bayesian Optimization/" alt="visitor"/>
    </p></center>


---

<br/>

# 📝 Bayesian Optimization

---

<br/>

베이지안 최적화는 Auto ML 분야에서도 Hyperparameter Optimization(하이퍼 파라미터 튜닝)에 대한 내용입니다. 

<br/>

## Auto ML

<br/>우선 Auto ML 분야에 대해 알아보겠습니다.

Auto ML은 '머신러닝으로 설계하는 머신러닝'으로, 학습 데이터가 정형 데이터일 때 Auto ML 기술을 적용하면 적은 노력으로 최적의 결과를 도출할 수 있습니다.

<br/>

Auto ML은 크게 3가지로 나뉩니다.

1. **Automated Feature Learning**
   - 입력 데이터 중 유의미한 피처를 추출하여 입력으로 사용하는 방법입니다.
   - 최적의 피처 추출 방법을 학습을 통해 찾을 수 있습니다.
2. **Architecture Search**
   - 학습을 통해 최적의 아키텍처를 설계하는 방법입니다.
   - 모델의 구조적 측면을 다루며 Darts 등이 이에 해당합니다.
3. **Hyperparameter Optimization**
   - 학습을 시키기 위해 필요한 하이퍼파라미터를 학습을 통해 추정합니다.

<br/>

Auto ML 패키지인 `PyCaret` 을 사용한다면 위 모든 과정을 한꺼번에 진행할 수 있지만, 이번 포스트에서는 **Hyperparameter Optimization** 위주로 다루겠습니다.

<br/>

---

<br/>

## Hyperparameter Optimization

<br/>

하이퍼 파라미터 튜닝은 학습을 수행하기 위해 사전에 설정해야 하는 파라미터의 최적값을 탐색하는 문제입니다. 

우선 파라미터는 모델 학습 시 필요한 여러 설정값입니다. 이 파라미터의 값에 따라 학습 결과에 큰 영향을 미칠 수 있습니다. 이러한 값들의 최적 조합을 뽑아내는 방법론이 Hyperparameter입니다.

Auto ML 을 적용하면 학습률(learning rate), 배치 크기(batch size) 등 학습에 영향을 주는 하이퍼파라미터들을 기존 수동적 조정에서 나아가 학습을 통한 최적의 하이퍼 파라미터를 추정할 수 있습니다.

먼저 기존 파라미터 탐색 방법을 소개하겠습니다.

<br/>

---

<br/>

### **Maual Search**

<br/>

수동적으로 파라미터 값을 탐색하는 방법입니다. 여러번의 탐색 과정 중 가장 좋은 결과값을 선택하는 방법으로 주관과 직관에 기반합니다.

이 방법은 실험을 통해 도출된 결과값이 **실제 최적값인지** 의문을 해소하기 어렵습니다.

![](http://research.sualab.com/assets/images/bayesian-optimization-overview-1/unluck-in-manual-search-process.gif)

<center> <small> <em> 출처는 포스트 하단에 일괄 명시했습니다. </em></small></center>

<br/>

나아가 한번에 하나의 파라미터를 추정하는 것이 아니라, 일반적으로 한번에 여러 종류의 파라미터를 동시에 탐색하는데 이러한 경우 파라미터 간 **상호 연관 관계를 무시할 수 없기에** 더욱 복잡한 연산을 수행해야 합니다. (예. Learning rate와 L2 정규화 계수)

<br/>

---

<br/>

### **Grid Search**

<br/>

Grid Search 는 Maual Search 의 단점을 보완하여 탐색 구간 내 추정하고자 하는 하이퍼파라미터 값 들을 일정한 간격을 두고 선정하여 가장 높은 성능을 발휘했던 하이퍼파라미터 값을 최종 선정하는 방법입니다.

![](http://research.sualab.com/assets/images/bayesian-optimization-overview-1/grid-search-process.gif)

<center> <small> <em> 출처는 포스트 하단에 일괄 명시했습니다. </em></small></center>

<br/>

물론 전체 탐색 대상 구간의 설정 방법, 간격의 길이 설정 방법 등 수동적인 요소는 남아있으나 균등하고 전역적인 탐색이 가능합니다. 그러나 추정하고자 하는 하이퍼 파라미터 개수를 늘리게 되면 탐색 연산 비용이 기하급수적으로 증가하게 됩니다. 

<br/>

---

<br/>

### **Random Search**

<br/>

Random Search는 Grid Search와 비슷한 맥락으로 탐색 대상 구간 내 하이퍼 파라미터 값들을 랜덤 샘플링을 통해 선정합니다.

![](http://research.sualab.com/assets/images/bayesian-optimization-overview-1/random-search-process.gif)

<center> <small> <em> 출처는 포스트 하단에 일괄 명시했습니다. </em></small></center>

<br/>

즉, 성능 함수의 최댓값이 예측되는 구간에 파라미터 조합을 랜덤으로 샘플링하여 최적 조합을 탐색하는 방법입니다.

Grid Search에 비해 불필요한 반복 수행 횟수를 줄일 수 있다는 장점이 있습니다.

<br/>

---

<br/>

### **정리**

<br/>

Random Search와 Grid Search는 하이퍼파라미터 값들의 **성능 결과에 대한 이전의 학습 결과가 반영되지 않습니다**. Maual Search의 경우 이전의 학습 결과를 바탕으로 수동으로 조합 값들을 조정했지만 Random Search와 Grid Search는 아직도 불필요한 탐색이 반복됩니다.

Bayesian Optimization방법은 이와 다르게 이전의 학습 결과를 반영하여 최적 조합을 탐색할 수 있습니다.

---

<br/>

## Bayesian Optimization

<br/>

베이지안 최적화 방법은 목적함수 `f`에 대해 함수값 `f`(`x`)를 최대로 만드는 최적해 `x`를 탐색하는 방법입니다.

이때 목적함수 `f`는 표현식을 명시적으로 알지 못하는 black box fuction 이며 하나의 함수값을 계산하는 데 오랜 시간이 소요된다고 가정합니다. 따라서 가능 한 적은 수의 `x` 후보에 대해서만 함수값을 연산하며 `f`를 최대로 만드는 최적해 `x`를 빠르고 효과적으로 탐색하는 것이 목적입니다.

이 목적함수 `f`는 모델의 성능 함수 `f`, `x`는 하이퍼 파라미터의 조합으로 생각하면 보다 쉽게 이해할 수 있습니다.

<br/>

목적함수 `f`를 추정하기 위해서는 **이전의 학습 결과가 반영된 모델**(Surrogate Model) 과 **다음 입력값 `x` 후보를 추천**해주는 함수(Acquisition Function) 이 필요합니다.

- **Surrogate Model** : 현재까지 조사된 입력값-함숫값 점들( (`x1`, `f(`x`1)`) (`x2`, `f(x2)` …), 을 바탕으로 `f`를 추정하는 확률모델로 **Gaussian Process** 가 주로 사용됩니다.
- **Acquisition Function** : `f`에 대한 Surrogate Model 결과를 바탕으로 다음 최적 입력값 `x`를 찾기 위해 다음 입력값 후보 `x i+1`를 탐색하는 함수입니다.

<br/>

이를 수도코드로 표현하면 다음과 같습니다.

```python
for i=1, 2, 3 … do
      surrogate model 의 확률적 추적 결과 바탕으로
      Acquisition Function를 최대화하는 입력값 후보xi+1설정
      
      f(xi+1) 계산
      surrogate model에 ( xi, f(xi+1) )추가하여 확률적 추적 수행
      
end
```

<br/>

먼저 Surrogate Model으로 가장 많이 사용되는 Gaussian Process에 대해 설명하겠습니다.

<br/>

---

<br/>

### **Gaussian Processes (GP)**

<br/>

보통의 확률 보델은 특정 변수에 대한 확률 분포를 표현합니다. 이에 반해 Gaussian Processes 는 함수에 대한 확률 분포를 나타내며 각 요소의 결합 분포가 가우시안 분포를 따른다는 특징이 있습니다.

<center><big> f(x)∼GP(μ(x),k(x,x′)) </big></center>

<br/>
위 식과 같이 평균 함수 μ와 공분산 함수 k를 사용하여 함수들에 대한 확률 분포를 표현합니다. 사진으로 각 함수의 역할을 자세히 살펴보겠습니다.

![](http://research.sualab.com/assets/images/bayesian-optimization-overview-1/bayesian-optimization-procedure-example.png)

<center> <small> <em> 출처는 포스트 하단에 일괄 명시했습니다. </em></small></center>

<br/>

위 사진은 t=2,3,4... 에 따라 GP의 연산 과정입니다. 초록색 음영은 Acquisition Function으로 다음 입력값 x를 탐색하고, 보라색 음영과 검정색 실선은 Gaussian Processes으로 f를 추정합니다. 검정색 점선은 실제 f 값인 미지의 함수입니다.

 t=2에서 Acquisition Function을 통해 다음 입력값 x(🔻으로 표시)를 탐색합니다. t=3에서 탐색된 입력값 x의 함수값(🔴)을 계산한 후 다음 입력값 x를 재 탐색합니다. 해당 과정을 반복하여 f의 최대값을 추정할 수 있습니다.

Gaussian Processes 부분을 자세히 살펴보겠습니다. 중앙 검정색 실선은 입력값 x에 대한 평균값 μ(x)이고, 검정색 실선을 둘러싼 보라색 음영은 x 위치 별 표준편차 σ(x)입니다.

σ(x)의 값을 살펴보면, 조사된 점(위 사진의 observation(x))의 값에서 멀어질수록 σ(x)이 크게 나타납니다(t=2에서 보라색 음영이 확장). 즉, 추정한 평균값 μ(x)의 **불확실성이 크다**는 뜻입니다.

거듭 연산이 진행될수록 `f`의 추정 결과가 압축됩니다(t=4 에서 보라색 음영의 축소). 즉, 조사된 점의 개수가 늘어날수록 평균값 μ(x)의 **불확실성이 감소**됩니다. 불확실성이 감소될수록 최적 입력값 x를 찾을 가능성이 높아집니다.

<br/>

---

<br/>

### **Acquisition Function**

<br/>

Acquisition Function은 `f`에 대해 확률 추정 모델(GP 등)의 결과를 바탕으로 t+1 번째 입력값 후보 `x i+1` 를 추천하는 함수입니다. 

이때 최적 입력값 `x`는 현재까지 조사된 점들 중 함수값이 최대인 점 근방에 위치하거나, 표준편차가 최대인 점 근방(불확실한 영역)에 위치할 가능성이 높습니다. 이 두 가지 경우를 각각 **exploitation**, **exploration** 전략이라 합니다. 이러한 경우의 수를 고려하여 `x i+1`을 탐색해야 합니다.

<br/>

1. **exploitation**
   - 현재까지 조사된 점들 중 **함수값이 최대인 점** 근방을 다음 차례에 시도합니다.
   - 함수값이 가장 큰 점 근방에서 실제 최적 입력값 `x`를 찾을 가능성이 높기 때문입니다.

2. **exploration**
   - 현재까지 추정된 목적 함수 상에서 **표준편차가 최대인 점 근방**을 다음 차례에 시도합니다.
   - 불확실한 영역에 최적 입력값 `x`이 존재할 가능성이 높기 때문입니다.

<br/>

Acquisition Function으로 가장 많이 사용되는 함수는 exploitation, exploration 전략 모두를 사용하는 **Expected Improvement** 입니다.

<br/>

---

<br/>

### **Expected Improvement(EI)**

<br/>

Expected Improvement는 현재까지 추정된 `f`를 바탕으로 어떤 입력값 `x`에 대해서 

1. 현재까지 조사된 점들의 **최대 함수값 `f(x+)` 보다 큰 함수값을 도출할 확률과**
2. **그 함수값과 최대 함수값 `f(x+)` 간 차이값**을

고려하여 `x`의 유용성을 반환합니다. 아래 그래프로 자세히 살펴보겠습니다.

![](http://research.sualab.com/assets/images/bayesian-optimization-overview-1/probability-of-improvement-in-gaussian-process-example.png)

<center> <small> <em> 출처는 포스트 하단에 일괄 명시했습니다. </em></small></center>

<br/>

위 그래프는 `x+` 이 계산된 `f`에 대해 **다음 후보 입력값인 `x1`, `x2`,`x3` 중 가장 유용한 값을 탐색**하는 과정입니다.

오른쪽의 초록색 음영은 **최대 함수값 `f(x+)` 보다 큰 함수값을 도출할 확률** `PI(x3)` 로,  다음 입력값으로 `x3`을 채택하는 것이 가장 유용하다고 판단됩니다.

따라서 `PI(x3)`값에 함수값 f(x3)에 대한 평균(검정색 실선에 해당하는 μ(x3)값)과 f(x3)-f(x+)을 가중하여 최종 EI를 계산합니다. 이 계산 과정을 통해 **f(x+)보다 큰 함수값을 도출할 수 있는 가능성** 뿐 아니라 **실제로 f(x3)값이 f(x+)보다 얼마나 더 큰 값**인지도 반영할 수 있습니다.

<br/>
수식으로 살펴보겠습니다.
<br/>

$$
\begin{align}
EI(x) & = \mathbb{E} [\max (f(x) - f(x^{+}), 0)] \\
      & = 
\begin{cases}
		(\mu(\boldsymbol{x}) - f(\boldsymbol{x}^{+})-\xi)\Phi(Z) + \sigma(\boldsymbol{x})\phi(Z) & \text{if}\ \sigma(\boldsymbol{x}) > 0 \\
    0 & \text{if}\ \sigma(\boldsymbol{x}) = 0 
\end{cases}
\end{align}
$$

$$
Z = 
\begin{cases}
    \frac{\mu(\boldsymbol{x})-f(\boldsymbol{x}^{+})-\xi}{\sigma(\boldsymbol{x})} & \text{if}\ \sigma(\boldsymbol{x}) > 0 \\ 
    0 & \text{if}\ \sigma(\boldsymbol{x}) = 0
\end{cases}
$$

<center> <small><em> EI 계산식 </em></small></center>
<br/>
- Φ : 표준정규분포의 누적분포함수(CDF)

- ϕ : 표준정규분포의 확률분포함수(PDF)

- ξ : exploration과 exploitation 간의 상대적 강도를 조절해 주는 파라미터

  - 클수록 exploration의 강도가 높아짐

  - 작을수록 exploitation의 강도가 높아짐

<br/>

---

<br/>

### **정리**

<br/>

지금까지 살펴본 **Bayesian Optimization** 의 수행 과정을 살펴보겠습니다.

```
1. 입력값, 목적 함수 및 그 외 설정값들 정의
    - 입력값 x: 학습률
    - 목적 함수 f(x) : 성능 함수 (e.g. 정확도)
    - 입력값 x의 탐색 대상 구간: (a,b).
    - 맨 처음에 조사할 입력값-함숫값 점들의 갯수: n
    - 맨 마지막 차례까지 조사할 입력값-함숫값 점들의 최대 갯수: N
2. 설정한 탐색 대상 구간내에서 처음 n개의 입력값들을 랜덤하게 샘플링하여 선택
3. 선택한 n개의 입력값 x1,x2,..,xn을 각각 학습률 값으로 설정
4. 딥러닝 모델을 학습한 뒤, 검증 데이터셋을 사용하여 학습이 완료된 모델의 성능 결과 수치를 계산
    - 이들을 각각 함숫값 f(x1),f(x2),...,f(xn)으로 간주
5. 입력값-함숫값 점들의 모음에 대하여 Surrogate Model로 확률적 추정 수행
6. 조사된 입력값-함숫값 점들이 총 N개에 도달할 때까지 반복
    - 기존 입력값-함숫값 점들의 모음에 대한 Surrogate Model의 확률적 추정 결과를 바탕으로, 입력값 구간 (a,b)(a,b) 내에서의 EI의 값을 계산
    - 그 값이 가장 큰 점을 다음 입력값 후보 xt+1로 선정
    - 다음 입력값 후보 xt+1을 학습률 값으로 설정하여 딥러닝 모델을 학습한 뒤, 검증 데이터셋을 사용하여 학습이 완료된 모델의 성능 결과 수치 계산 → f(xt+1)값
    - 새로운 점 (xt+1,f(xt+1))을 기존 입력값-함숫값 점들의 모음에 추가
    - 갱신된 점들의 모음에 대하여 Surrogate Model로 확률적 추정을 다시 수행
7. 총 N개의 입력값-함숫값 점들에 대하여 확률적으로 추정된 목적 함수 결과물을 바탕으로, 평균 함수 μ(x)을 최대로 만드는 최적해 x∗를 최종 선택
8. 해당 x∗ 값을 학습률로 사용하여 딥러닝 모델을 학습하면, 일반화 성능이 극대화된 모델을 얻을 수 있음
```

<br/>

----

<br/>

# **📗 정리**

<br/>

이상으로 Bayesian Optimization에 대해 살펴보았습니다. 실제 적용에 대해서는 데이콘의 심리성향 예측 대회 데이터를 바탕으로 다루겠습니다. 아래 링크에서 확인할 수 있습니다.

- [데이콘 심리성향 예측 대회] AUTO ML - 베이지안 최적화 (Bayesian Optimization) 바로가기 [링크](jihyun22.github.io/데이콘리뷰/psychology-02/)

<br/>

---

<br/>

## 참고자료

<br/>

- Shahriari et al., Taking the human out of the loop: A review of bayesian optimization.
  - [Shahriari, Bobak, et al. “Taking the human out of the loop: A review of bayesian optimization.” Proceedings of the IEEE 104.1 (2016): 148-175.](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf)
- Brochu et al., A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning.
  - [Brochu, Eric, Vlad M. Cora, and Nando De Freitas. “A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning.” arXiv preprint arXiv:1012.2599 (2010).](https://arxiv.org/pdf/1012.2599.pdf?bcsi_scan_dd0fad490e5fad80=fwQqmV5CfHDAMm8dFLewPK+h1WGiAAAAkj1aUQ%3D%3D&bcsi_scan_filename=1012.2599.pdf&utm_content=buffered388&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer)
- Bengio et al., Practical recommendations for gradient-based training of deep architectures.
  - [Bengio, Yoshua. “Practical recommendations for gradient-based training of deep architectures.” Neural networks: Tricks of the trade. Springer, Berlin, Heidelberg, 2012. 437-478.](https://arxiv.org/pdf/1206.5533.pdf)
- Goodfellow et al., Deep learning.
  - [Goodfellow, Ian, et al. Deep learning. Vol. 1. Cambridge: MIT press, 2016.](https://www.deeplearningbook.org/)
- Bergstra and Bengio, Random search for hyper-parameter optimization.
  - [Bergstra, James, and Yoshua Bengio. “Random search for hyper-parameter optimization.” Journal of Machine Learning Research 13.Feb (2012): 281-305.](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
- Fernando Nogueira, bayesian-optimization: A Python implementation of global optimization with gaussian processes.
  - https://github.com/fmfn/BayesianOptimization
- Hunting Optima, Expected Improvement for Bayesian Optimization: A Derivation.
  - http://ash-aldujaili.github.io/blog/2018/02/01/ei/

<br/>













































