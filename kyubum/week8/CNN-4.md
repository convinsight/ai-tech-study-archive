---
title: 부스트 캠프 ai tech 8주 4일차 CNN visualization
date: 2022-03-10 10:15:57
tags:
- DeepLearning
- week8
- CV
- CNN Viz
categories:
- boostcamp
- week
widgets: null
mathjax: true
---
***
## CNN visualization 의 필요성
* 기존 DeepLearning 모델은 내부를 볼 수 없는 시스템(Black Box)라고 여겨졌다
  * 실제로는 내부 Parameter 값을 볼 수 있지만 weight들로 이루어진 Metrix만 존재하기 때문에 해석하기 힘들다
* CNN visualization에서는 weight들을 시각화 해서 설명가능하게 만들어 준다
  * filter들을 시각화 함으로써 어떤방식으로 동작하는지 설명이 가능해진다
  * weight의 GradCAM을 통하여 어떤부분에 모델이 집중하는지 보여주면서 왜 잘 동작했는지, 동작하지 않았는지를 더 쉽게 설명이 가능하다
  * 시각화된 결과를 기반으로 추가적인 성능 향상을 위한 가설을 세울 수 있다

# CNN Visualization
* CNN Visualization에는 다양한 방법들이 존재한다
* 이 글에서는 Model behavior analysis와 Model Decision analysis부분에 대해서 다룬다

<center>

<img src="/img/CNNViz.PNG" alt="" width="600px"/>

</center>

# Model behavior analysis
* 모델 자체의 행동에 집중하여 분석하는 기법

## Embedding feature analysis
High Level의 Layer에서 얻어지는 Feature를 분석하는 기법이다

### Nearest Neighbors in Feature Space
* Nearest Neighbors를 이용한 모델 시각화이다
* Neural Networks를 이용하여 High Level의 Feature를 뽑고 이를 이용하여 DB를 생성한다
* test Data를 Model에 넣어서 kNN으로 모델이 생성한 High dimensional Feature Space를 확인한다
* 예제를 보고 판단하기 때문에 전체적인 부분을 확인하기는 힘들다

### t-SNE
* 위의 모델이 생성한 High demensional Feature Space를 Low dimensional Space로 변화시켜서 시각화하는 방법
* Feature의 전체적인 그림을 그려주어서 Feature Space를 어느정도 이해할 수 있도록 도와주는 역할을 한다

<center>

<img src="/img/tSNE.PNG" alt="" width="600px"/>

</center>

## Activation investigation
Mid ~ High Level Layer에서 이루어지는 Feature 분석 기법이다

### Layer Activation
* mid to high level hidden unit의 행동을 파악해보는 기법
* 특정 Layer의 특정 Node를 가공하여 어느부분을 집중적으로 보는 node인지를 masking한다

### Maximally activating patches
* Hidden Node 별로 가중치가 가장 높은 부분을 뜯는 기법
* 국부적인 부분에 적합하여 Mid Level Feature에서 사용한다

### Class visualization
* 예제 데이터를 사용하지 않고 네트워크의 parameter로 이미지를 시각화 하는 방법

<center>

<img src="/img/classviz.PNG" alt="" width="600px"/>

</center>

* 특정 class에 대한 네트워크의 예상치를 확인하는 방법이다
* 이것을 보고 주변객체와의 연관성 등도 파악이 가능하며 데이터의 편향성이 존재하는지도 파악 할 수 있다
* 특정 이미지를 넣어서 확인하는 것이 아닌 dummy 이미지를 넣어서 확인한다

## Model Decision analysis
* 모델의 특정 입력에 대해서 경향에 집중하여 분석하는 기법

### Occulsion map
* 특정 부분에 Occusion patch를 이용하여 가린 이미지들로 뽑은 Score바탕으로 heapmap을 구성하는 방법
* 특정 이미지의 스코어 영향을 미치는 영역을 파악할 수 있다.

### via Backpropatation
* 특정 이미지에 대해서 classification하는데에 영향을 미친 부분을 heapmap으로 표시하는 기법
1. 특정 이미지에 대한 class의 스코어를 얻는다
2. Backpropagation을 통해서 입력 이미지의 Gradient를 구한다
3. 얻어진 Graident의 magnitude를 구한다
   * 얼마나 큰 영향을 끼쳤는지가 중요하기 때문에 부호를 제거한다
4. 해당 map을 시각화

### Class Activation Mapping
* 특정이미지에 대해서 어떤 결과가 나왔고 어떤부분을 참조하였는지를 보여주는 기법
* Global Average Pooling + FC Layer가 있는 모델에서만 사용이 가능하다

### Grad CAM
* CAM과 같이 특정이미지에 대해서 어떤 결과가 나왔고 어떤부분을 참조하였는지를 보여주는 기법
* CNN Backbone이기만 하면 어떤 모델이든지 사용이 가능하다

<center>

<img src="/img/GradCAM.PNG" alt="" width="600px"/>

</center>

### reference
* [Naver Connect Boostcamp - ai tech](https://boostcamp.connect.or.kr/program_ai.html)