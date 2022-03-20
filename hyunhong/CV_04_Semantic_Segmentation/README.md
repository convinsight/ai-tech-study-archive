# Semantic Segmentation

#### 1. Data augmentation

1.  What is semantic segmentation?
   - pixel 단위로 image classification을 수행
   - class만 구분하고 객체 구분은 instance segmentation
2. Where can semantic segmentation be applied to?
   - medical, autonomous driving...

#### 2. Semantic segmentation architectures

1.  Fully Convolutional Networks (FCN)
   - End to End semantic segmentation
     - 입력부터 출력까지 미분 가능한 neural network
   - 임의의 크기의 이미지에 호환
   - vs Fully connected layer
     - Fully connected layer
       - fixed dimensional vector I/O
       - activation map이 출력되면 flattening하여 공간을 고려하지 않은 하나의 벡터로 합쳐짐
     - Fully convolution layer
       - activation map(tensor) I/O
       - spatial coordinate가 보존
       - 1x1 convolution
   - channel axis로 flattening
     - 각 층을 fully-connected하게 만들 수 있음
     - 1x1 conv와 같음
   - 작은 score map
     - stride, pooling layer에 의해서 필연적인 저해상도
   - Upsampling
     - 작게 만들어 receptive field는 키우고 upsampling을 통해 복구
     - Transposed convolution
       - input과 filter를 곱해서 더하는 방식으로 overlap 부분이 발생
     - Upsample and convolution
       - 일부만 overlap이 발생하는 문제를 피하고 일괄적인 convolution 수행
     - 잃어버린 정보를 얻는 일은 쉬운 일이 아님
     - Low layer
       - detail, local
     - high layer
       - semantic, holistic, global
     - 높은 layer에 있던 activation map을 upsampling하고, 중간 층의 layer 또한 upsampling 해서 가져옴
       - activation map을 형성
2.   Hypercolumns for object segmentation
   - 2015년에 나온 FCN과 유사한 실험
     - 강조 포인트가 낮은 layer와 높은 layer의 융합에 있음
   - End to End model이 아님
3.  U-Net
    - Fully convolutional
    - skip connection을 통해 더 정교한 결과물
    - upsampling을 단계적으로 진행
      - 대칭하는 위치의 layer들을 가져와 fusion하여 특징을 찾음
    - Contracting path
      - 채널 수가 늘어나며 scale 감소
      - downsampling과 upsampling이 잦게 발생하므로 홀수가 나오면 안 됨
    - Expanding path
      - 채널 수가 줄어들며 scale 증가
4.  DeepLab
    - Dilated convolution
      - receptive field가 커짐
