---
title: 부스트 캠프 ai tech 8주 1일차 CNN architectures
date: 2022-03-07 10:41:40
tags:
- DeepLearning
- week8
- CV
categories:
- boostcamp
- week
widgets: null
mathjax: true
---
***
## Image Classifier
* 이미지를 분류하는 기본적인 모델을 말한다
* 충분한 데이터가 존재한다면 모든 분류문제는 K-Nearest Neighobors로 해결이 가능하다.
  * 영상 분류 문제를 검색 문제로 바꿔서 해결이 가능하다 
  * 하지만 데이터는 너무 많고, 우리의 머신 리소스는 한정적이기 때문에 모든 문제를 해결하기에는 적합하지 않다
  * 데이터 수에 비례해서 시간복잡도가 증가한다
* 초창기에는 single Fully Connected Layer를 이용해서 분류 문제를 해결하려 했다
  * 레이어층이 적다보니 평균적인 이미지에서 벗어나면 잘 작동하지 않았다(test 성능이 좋지 않았다)
* 이미지를 전체적으로 보는것이 아닌 부분적으로 파라미터 연산을 하는 Locally Connected Layer가 등장하였고 Convolution Layer의 전신이 되었다

## Data Augmentation
* 우리가 모델을 학습시킬때 사용하는 Data는 실제 전체 데이터에서 샘플링한 극히 일부의 데이터이다. 또한 데이터를 제작한 사람의 주관또한 들어가 있을 가능성이 존재하기 때문에 Train Data의 분포가 실제 데이터와 일치한다고 보기 힘들다. 이를 보안하기 위해서 Data의 분포를 다양하게 만드는 방법을 Data Augmentation이라고 한다.
* 기본적인 Augmentation
  * 밝기, 채도, 명암 조절
  * Random Crop, Filp, Rotate
  * Affine Transform : 기하학적 변환
    * warp로 시작하는 함수
* 특수한 Augmentation
  * CutMix : 두개의 사진을 잘라서 합치는 Augmentation. 라벨값 또한 비율에 맞게 조절한다  

<center>

<img src="/img/Cutmix.PNG" alt="" width="500px"/>

</center>

  * MixUp : 두개의 사진을 Alpha값의 조절을 통한 픽셀을 합치는 Augmentation. 라벨값 또한 비율에 맞게 조절한다  

<center>

<img src="https://blog.airlab.re.kr/assets/images/posts/2019-11-23-mixup/img_01.png" alt="" width="500px"/>

</center>

## Transfer Learning
* 기존에 학습시킨 네트워크를 이용하여 새로운 Task를 해결하는 모델에 재학습 시키는 방법
* 학습시키는 Dataset이 비슷한 분포를 가지고 있어야 더욱 잘 학습 된다.
* Layer Freeze
  * 특정한 Layer의 Parameter를 Freeze시켜서 고정시키고 나머지 Layer로만 학습시키는 방법
  * 데이터의 양이 적을때 효과가 좋다
* Fine Tuning
  * 새로 추가된 Layer의 Learning Rate와 기존 CNN 부분의 Learning Rate를 다르게(기존 부분을 더 작게) 설정하여 Tuning하는 방법
  * 어느정도 데이터가 존재할때 효과가 좋다

## Knowledge distillation
* Teacher Student Learning
* 큰 모델에서 학습한 weight를 작은 모델에도 비슷하게 작동하게 전달하는 기법
  * 최적화방면에서도 많이 이용된다.

### reference
* [Naver Connect Boostcamp - ai tech](https://boostcamp.connect.or.kr/program_ai.html)
* [Mixup: BEYOND EMPIRICAL RISK](https://blog.airlab.re.kr/2019/11/mixup)