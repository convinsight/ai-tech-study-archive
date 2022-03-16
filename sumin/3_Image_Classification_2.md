# Image Classification 2



## Introduction



- 이번 강의에서는 1강 Image Classification에 이어서 대표적인 CNN 모델들에 대해 배웁니다

- 먼저 VGGNet과 비슷한 시기에 등장한 GoogLeNet을 시작으로,  
  지금도 많이 쓰이고 있는 ResNet에 공부하고 실습을 진행합니다

- 이 외에도 추가적으로 몇가지 CNN 모델들에 대한 소개를 합니다

- 끝으로 1강과 3강까지 다룬 4가지 모델 (AlexNet, VGGNet, GoogLeNet, ResNet)에 대하여  
  메모리 측면과 계산 효율 관점에서 비교 분석을 합니다





## 1. Problems with deeper layers



### 1.1 Going deeper with convolutions

- Network 가 더 깊어지고 (deeper), 넓어지는 (wider) 트렌드가 이어짐



- Deeper networks learn more powerful features
  - Larger receptive fields
  - More capacity and non-linearity



- 하지만, 여러 문제가 존재하게 되었다



### 1.2 Hard to optimize

- Deeper networks are harder to optimize



- Gradient Vanishing 문제
- Gradient Exploding 문제



- Computationally complex



- Degradation 문제





## 2. CNN architectures for image classification 2



### 2.1 GoogLeNet

- Inception Module

  ![image-20220316030954686](https://s2.loli.net/2022/03/16/wPBncfOGeqxjYJR.png)

  - Apply multiple filter operations on input activation from the previous layer

    - 1x1, 3x3, 5x5 convolution filters
    - 3x3 pooling operation

    

    - 하나의 Layer 에서 다양한 크기의 convolution filter 를 사용하여,  
      여러 측면으로 activation 을 관찰
    - depth 가 아닌 수평 확장

    

  - Concatenate all filter outputs together along the channel axis

    - 결과 activation map 을 channel 축으로 concatenate 하여 다음 block 으로 전달

    

  - 1x1 convolutions 을 'bottleneck' layer 로 사용하여,  
    channel 의 수를 줄임

    - network size 를 줄이는 효과
    - max pooling 에도 적용하여 channel 의 수를 줄여주었다



- 1x1 convolutions

  ![image-20220316031153837](https://s2.loli.net/2022/03/16/hQVg94fHPDmYldA.png)

  - filter 의 갯수 (m=2) 만큼,  
    output channel 의 갯수를 줄이게 된다
  - 공간의 크기는 변하지 않으며,  
    pixel 독립적으로 channel 의 수만 줄어들게 된다



- Overall architecture

  ![image-20220316031526271](https://s2.loli.net/2022/03/16/NdLpqOY1wtzCEs6.png)

  - Stem network

    - vanilla convolution networks
    - 일반적인 CNN 모습

    

  - Stacked inception modules

    - inception module 이 여러 개 쌓여있음

    

  - Auxiliary classifiers

    - 보조 분류기
    - network 가 깊어지면 backpropagation gradient 가 소실되기에,  
      보조 분류기를 통해 이를 보완

    

  - Classifier output

    - a Single FC layer
    - 하나의 FC layer 를 통해 softmax classification score 를 출력



- Auxiliary classifier

  ![image-20220316031827937](https://s2.loli.net/2022/03/16/kEOoRBXf6ZjJl4F.png)

  - The vanishing gradient problem is dealt with by the auxiliary classifier
    - vanishing gradient problem 를 해결하기 위함
  - Injecting additional gradients into lower layers
  - Used only during training, removed at testing time
    - 학습 시에만 사용하며,  
      test 시에는 맨 마지막 결과 (classifier)만 사용하고, Auxiliary classifier 는 사용하지 않는다





### 2.2 ResNet

- ResNet

  - ResNet 은 최초로 100 개 이상의 Layer 를 쌓으며,  
    더 깊은 Layer 로 더 좋은 성능을 이끌어낸 모델

  - 인간의 능력을 뛰어넘으며 2015년 ImageNet 대회에서 1등 차지

  - ImageNet Classification 뿐아니라, localization, Detection, Segmentation 도 1등 차지




- Revolutions of depth
  - Building ultra-deeper than any other networks
  - What makes it hard to build a very deep architecture
    - **Degradation problem**
    - Degradation problem 으로 layer 가 깊어져도 성능이 향상되지 않던 한계를 극복



- Degradation problem
  - As the network depth increases, accuracy gets saturated -> degrade rapidly
  - 모델의 parameter 가 많아지면 Overfitting 에 취약할 것이라고 생각했었지만,  
    더 깊은 층 (56-layer) 의 training, test error 가 모두 더 얕은 층 (20-layer) 의 error 보다 안좋은  
    성능을 보여주며 Overfitting 문제가 아닌 **degradation problem** (optimization) 문제라는 결론을 냄
    - cf) Overfitting 이 문제였다면, 더 깊은 층의 training error 는 더 좋은 성능을 내지만,  
      test error 에서 더 안좋은 성능을 보여주었어야한다

![image-20220316133849697](https://s2.loli.net/2022/03/16/ZJEHjLWpRbcy2GK.png)



- **Hypothesis**

  - Plain Layer

    - As the layers get deeper, it is hard to learn good $H(x)$ directly
    - $H(x)$ 라는 mapping 을 학습할 때에,  
      layer 를 높게 쌓아서 곧바로 $x$ 에서 $H(x)$ 의 관계를 학습하려고하면,  
      복잡하기에 학습하기 어렵다

    

  - Residual block

    - 입력으로 주어진 $x$ (identity) 외의 잔여 부분 (residual) 만 모델링하여,  
      학습하게끔 변경
    - Target function : $H(x) = F(x) + x$
    - Residual function : $F(x) = H(x) - x$



- **Solution : Shortcut connection**
  - skip connection 이라고도 한다
  - 기존의 gradient 가 vanishing 되더라도, shortcut connection (identity) 에 의한 gradient 가 남아있기에,  
    gradient vanishing 문제를 해소할 수 있게됨

![image-20220316134559604](https://s2.loli.net/2022/03/16/LvnfwCXk3gsYQ9G.png)



- Analysis of residual connection
  - gradient 가 지나갈 수 있는 2^n^ 개의 input, output path 가 생성된다
    - 다양한 경우의 수를 갖는 경로를 통해서 복잡한 mapping 을 학습할 수 있게 된다는 분석
    - residual block 이 하나 추가될 때마다 경로 수가 2배씩 증가
  - Residual networks have $O(2^n)$ implicit paths connecting input and output,  
    and adding a block doubles the number of paths

![image-20220316134944556](https://s2.loli.net/2022/03/16/pDWKN1QjCtbxHc9.png)



- Overall architecture

  ![image-20220316142440075](https://s2.loli.net/2022/03/16/Fy6TRq9WgGQDCwx.png)

  - 시작 part

    - 7x7 Conv layer 1개
    - He initialization
      - ResNet 에 적합한 initialization
      - He 가 아닌 일반적인 initialization 을 사용하면,  
        skip connection 과정에서 처음부터 더해지는 값이 커지게된다

    

  - Residual Block part

    - Stack residual blocks

    - Every residual block has two 3x3 conv layers

      - residual block 내부의 layer 는 모두 3x3 conv 사용
      - 상대적으로 연산이 빠른 이유

    - Batch norm after every conv layer

      

    - Doubling the number of filters and spatially down-sampling by stride 2 instead of spatial pooling

      - residual block part 가 바뀔때마다,  
        down sampling 을 통해 공간상의 크기는 절반으로 줄고, 채널 수는 두 배씩 증가

    

  - 최종 출력

    - Only a single FC layer for output classes
    - avg pool 을 적용한 후, 하나의 FC layer 로 구성







### 2.3 Beyond ResNet

- DenseNet

  ![image-20220316032335360](https://s2.loli.net/2022/03/16/cXNijnZf8K3PV59.png)

  - channel 축으로 concatenate 진행

    - 바로 직전 block 의 입력뿐만 아니라, 이전의 정보도 반영
    - 상위 layer 에서도 하위 layer 의 정보를 참조하게 됨

  - Alleviate vanishing gradient problem

  - Strengthen feature propagation

  - Encourage the reuse of features

    

  - In ResNet, we added the input and the output of the layer element-wisely

  - In the Dense blocks, every output of each layer is concatenated along the channel axis

    - channel 이 늘어나서 메모리와 computational complexity 가 증가한다는 단점이 있지만,  
      feature 의 정보를 그대로 보존한다는 장점이 있다

    









