# CNN Visualization

convolution network의 시각화

#### 1.   Visualizing CNN

1.  What is CNN visualization?
   - 해석이 잘 되지 않는 CNN의 내부 구조를 가늠해보기 위함
   - 성능 향상을 위한 분석
   - debugging tool과 같은 역할
   - deconvolution을 이용한 연구
     - ZFNet
       - low level: line, block
       - high level: feature
2. Vanilla example: filter visualization
   - filter visualization
     - 1st conv.layer
       - color
       - angle
       - detail
     - 뒤쪽 layer는 차원 수가 높아 직관적으로 해석할 수 없음
3. How to visualize neural network
   - model의 입력 또는 출력에 집중하는 방법들

#### 2. Analysis of model behaviors

1.  Embedding feature analysis
   - Nearest neighbor search
     - DB에서 가장 유사한 query data를 검색
     - clustered concept로 구분
     - 각 vector에 해당하는 영상 존재
     - fc layer를 없애고 특징 추출 layer를 형성
     - 특성들을 DB에 저장
   - dimension reduction
     - t-SNE
   
2. Activation investigation
   - layer activation
   
     - hidden node
   
   - Maximally activation patches
   
     - hidden node가 무엇을 찾는 detector인지 찾음
   
     1. channel의 특정 layer를 선택
     2. data를 입력하고 선택한 layer의 activation 저장
     3. 해당 channel의 activation 중 가장 큰 값을 도출한 receptive field를 crop
   
   - Class visualization
   
   $$
   I = argmaxf(I) - Reg(I)\\
   e.g.\space\space\space\space Reg(I) = \lambda||I||^2_2
   $$
   
   1. get prediction score
   2. backpropagation
   3. update image

#### 3. Model decision explanationSemi-supervised

1. Saliency test
   - Occlusion map
   
     - 다양한 영역을 가려보고 판별에 중요한 부분을 찾는 기법
   
     1. 이미지를 가려서 입력
     2. 원래 class를 반환할 확률 구함
   
     - 진할 수록 영향이 큼
   
   - via Backpropagation
   
     - classification 해보고 heat map을 나타내는 기법
     - 밝은 영역이 영향을 준 부분
   
     1. 입력 영상을 넣고 class score 획득
     2. gradient의 절대적 크기를 구한 magnitude map 구현
        - 여러 회 수행하여 축적하는 것도 가능
   
2. Backpropagate features
   - Rectified unit
     - ????
   - guided backpropagation
     - forward 할 때 긍정적인 영향을 끼친 양수를 참조
     - backward할 때 gradient를 통해 강화하는 activation 참조
   
3. Class activation mapping(CAM)
   - conv block을 통과한 후 fc layer를 통과하지 않고, Glabal average pooling(GAP) layer를 지남
     $$
     S_c = \sum_k w^c_k \sum_{(x , y)}f_k(x, y)
     $$
   
   - 효과가 매우 좋음
   
   - 마지막 layer가 GAP layer로 변경되며 모델 구조가 변해 성능 저하 가능
   
     - ResNet이나 GoogLeNet은 마지막 구조가 하나의 fc layer와 average pool을 사용해 CAM 적용 용이
   
   - Grad-CAM
   
     - CNN backbone만 가지고 있으면 Task에 무관한 적용
   
       - important weigt 구하기
         $$
         \alpha ^c_k = \frac{1}{Z}\sum_i \sum_j \frac{\delta y^c}{\delta A ^k_{ij}}
         $$
         
       - 결합
         $$
         L^c_{Grad-CAM} = ReLU(\sum_k\alpha^c_kA^k)
         $$
     
   - Guided Grad-CAM
   
     - Grad-CAM은 lough하고 smooth
     - Guided Backprop은 sharp하지만 class 구분성이 떨어져 전반적인 response
     - 두 기법을 곱함
   
   - GAN dissection
