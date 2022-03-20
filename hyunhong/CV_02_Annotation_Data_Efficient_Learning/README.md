# Annotation Data Efficient Learning

머신러닝 알고리즘은 데이터가 아무리 많아도 더 요구하지만
이를 만족시키는 Dataset을 만들기는 매우 어려움

Annotation data의 양과 효율을 올리는 것이 과제

#### 1. Data augmentation

1.  Learning representation of dataset

   - 실제 세상의 데이터는 대부분 biased, 편향된 이미지
     - 촬영 장비를 통해 얻은 획득된 데이터는 현실과의 거리가 있음
   - Dataset이 세상을 모두 담을 수는 없음

2. Data augmentation

   - 주어진 데이터를 풍부하게 만들기 위한 기법
     - `Crop`, `Shear`, `Brightness`, `Perspective`, `Rotate` ···
     - `OpenCV`와 `numpy`에 많은 방법이 포함되어있고, 이를 현실 분포와 비슷하게 만드는 것이 목표

3. Various data augmentation methods

   - Brightness adjustment

     - tensor 값을 높이거나 낮춤

   - Rotate, flip

     - OpenCV 

   - Crop

   - Affine transformation

     - = Shear
     - 길이의 비율과 선, 평행관계는 유지한 상태로 변경
     - 점 3개를 포함하는 대응 쌍으로 입력하면 대응되는 위치로 픽셀을 옮김

   - CutMix

     - 잘라서 합치고 라벨도 비율대로 합침

     - 의외로 손쉽게 성능 향상이 일어남

       *mask 분류때는 어려웠던 것으로 보아 전혀 다른 label을 분류할 때 효과가 좋을것 같음

4.  Modern augmentation techniques

   - RandAugment
     - Random Augmentation
     - 어떤 augmentation을 얼마나 적용할 지
     - Policy: 적용할 Augmentation의 set

#### 2. Leveraging pre-trained information

높은 품질의 Dataset을 구성하는 것 자체가 어려움

1.  Transfer learning

   - 학습된 사전지식을 활용해 새로운 기능에서도 쉽게 학습할 수 있음
   - 한 Dataset에서 배운 지식을 다른 Dataset에서 활용하는 기능
     - 공통된 지식이 많을 것이라는 가정에서 시작됨
   - 접근법
     1. Convolution Layer는 고정시키고 마지막 Fully Connected Layer(fc)만 변경 및 학습하여
        output을 변경하여 원하는 class로 학습
        - 데이터 양이 정말 적을 때 사용
     2. Convolution Layer도 Low learning rate를 부여하는 방법
        - fine tuning
        - 데이터가 상대적으로 적절하게 존재할 때 활용
        - 속도는 조금 늦어지는 단점

2. Knowledge distillation

   - Teacher-student learning
   - model 압축에 유용하게 사용
   - pseudo-labeling에 활용 가능
   - 절차
     1. Pre-trained  Model와 Student Model을 준비하고 같은 input을 동시에 feed
     2. Output을 KL div.Loss를 통해 비교하고 Student Model을 학습시킴
   - unsupervised learning
   - 임의의 데이터를 사용할 수 있음

   > **Softmax**
   >
   > 0 또는 1의 hard labeling이 아닌 prediction의 개념을 위해 0과 1 사이의 값을 출력하는 함수
   >
   > **temperature**
   >
   > soft labeling을 돕기위해 입력 값을 극단적으로 벌려주는 역할
   > 전반적인 분포 파악에 유리

   - Sementic Information이 중요한 것이 아니라, 개형을 따라가는 것
   - Distillation Loss
     - KLdiv(Soft label, Soft prediction)
     - teacher와 student 사이의 차이를 측정하고 따라하도록 만듦
   - Student Loss
     - student network와 true label이 일치하도록 만듦

#### 3. Leveraging unlabeled dataset for training

1. Semi-supervised learning
   - unlabeld data를 효율적으로 사용하는 방법
     - 전체의 수 많은 unlabeled data와 labeled data를 함께 활용
     - unsupervised + supervised
   - 절차
     1. Label data로 model pre-training
     2. pseudo-labeled data 생성
     3. labeled dataset과 합쳐 새로운 model 학습
2. Self-training
   - noisy student
     1. ImageNet labeled data로  teacher model학습
     2. teacher모델이 unlabeled data를 pseudo labeled data로 만듦
     3. student model을 학습시킴
     4. student model이 teacher model로 대체
     5. 2~4 반복
   - 기존의 한계를 극복한 패러다임
