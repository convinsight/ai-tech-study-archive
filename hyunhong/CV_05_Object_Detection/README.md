# Object detection

#### 1.  Object detection

Semantic → Instance → Panoptic Segmentation

1.  What is object detection?
   - Classification과 Box localization의 복합 기술
     - 복합기술을 필요로 함
2. What are the applications of object detection?
   - 자율주행
   - OCR(Optical Character Recognition)

#### 2. Two-stage detector (R-CNN family)

0. Traditional methods
   - 경계를 찾는 알고리즘
   - Selective search
     1. over-segmentation 색깔로 구분
     2. 기준에 따라 비슷한 영역끼리 합침
     3. segmentation 단위로 bounding box로 나눠 후보군으로 사용

1.  R-CNN

   1. Input image
   2. selective search 등으로 region proposal을 구함
      - 별도의 hand designed 기법으로 한계가 생김
   3. warping한 이미지를 학습된 CNN 사용
   4. classifier는 svm 등의 기법

2.  Fast R-CNN
   - 영상 전체에 대한 정보를 한 번에 추출하여 재활용하여 여러 object detection

   1. Conv. feature map 추출
      - fully convolution한 network는 input size 무관하므로 warping 생략
   2. (1)에서 Region of Interest(RoI) feature를 추출
   3. Pooling 및 resampling된 feature에서 Classification과 bbox regrossion 수행

   - Feature 재활용을 통해 기존의 R-CNN에 비해 최대 18배 빠른 R-CNN
   - 별도의 algorithm을 사용하는 것은 여전히 성능의 한계를 만듦

3.  Faster R-CNN
   - region proposal을 neural network에 의존하도록 바꾼 최초의 end to end object detection
   - IoU
     - Intersection over Union
       - 두 영역의 교집합/합집합
   - Region Proposal
     - anchor box
       - 비율과 scale이 다른 박스를 생성
       - 정사각형, 직사각형 후보군
     - Region Proposal Network
       - 기존의 hand designed algorithm 대체
   - procedure
     - sliding window 방식으로 순환
     - conv feature map에서 각 지점에서 k개의 anchor box
       - 256 dimension
       - 2k score class layer
         - object vs non-object
       - 4k coordinates regression layer
     - non-Maximum Suppression
       - 그럴듯한 object bbox만 남겨두고 제거

#### 3.  Single-stage detector

0. Comparison with two-stage detectors
   - 정확도를 조금 포기하더라도, 속도를 확보하여 Real Time Detection 하는 것에 목표
   - Region Proposal을 기반으로한 RoI pooling을 하지 않고 곧바로 Box repression과 classification을 수행
     - 빠른 작업 시간

1. YOLO
   - You Only Look Once
   - Input image를 S x S grid로 나누고, grid에 대해 B개의 box(4개의 좌표와 class)를 예측
   - NMS를 통해 box 반환
   - 학습 방법은 Fast-RCNN과 동일
   - output
     - 30channel
       - 5(x, y, w, h, obj score)*B(2) + C(20 classes)
2. Single Shot MultiBox Detector(SSD)
   - multi-scale object를 더 잘 처리하기 위해 중간 feature map을 각 해상도에 적절한 bounding box로 출력할 수 있도록 multi-scale 구조를 형성
   - VGG Backbone

#### 4. Single-stage detector vs. two-stage detector

- single stage 기법은 RoI pooling이 없어 모든 영역의 loss가 개선되고 일정 gradient가 발생
- 일반적인 영상은 background가 더 넓음
  - 유효한 정보가 없으면서 많은 영역과 개수로 class imbalance를 만듦

1. Focal loss
   - cross entropy의 확장
   - 잘 맞춘 경우 loss를 더 낮게, 오답일 때는 더 sharp한 gradient 발생
2. RetinaNet
   - Feature Pyramid Networks + class/box prediction branches
   - U-Net과 유사한 구조
   - 더하기로 fusion
   - class subnet과 box subnet이 따로 구성되어 진행

#### 5. Detection with Transformer

1. DETR
   - Transformer를 object detection에 적용
   - CNN feature와 multi dimension으로 표현한 encoding을 쌍으로 input token을 만듦
   - transformer encoder를 거쳐 추출된 특징을 decoder로 입력
   - 학습된 positional encoding인 object query를 활용해 질의
     - 하나의 위치에서 N개의 object가 발생 가능함을 미리 입력
   - 정보가 parsing되어 출력
     - class와 box 정보가 있을 때만 출력
2. Further
   - 물체의 중심점을 찾는 등의 방법
