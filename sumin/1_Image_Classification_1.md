# Image Classification - 1



## Introduction



- 우리는 오감 중 특히 시각에 의존하여 사물을 바라보고 이해하며 살아가고 있습니다
- 동일한 프로세스를 컴퓨터에 적용한 컴퓨터 비전입니다
- 본 강의에서는 컴퓨터 비전 (CV)의 첫 시간으로 CV에 대해 짧게 소개하고,  
  CV에서 가장 기본적인 task, image clasiification을 소개합니다
- Image Classification은 사진이 주어졌을 때 특정 카테고리로 분류하는 task입니다
- 이번 강의에서는 먼저 기존의 머신러닝과 구분되는  
  딥러닝을 사용한 Image classification 의 특징에 대해서 배웁니다
- 다음으로 대표적인 CNN 모델인 AlexNet을 배우고 이에 대한 실습을 진행합니다
- 끝으로 가장 유명한 classification 모델 중 하나인 VGGNet에 대해 배웁니다.





## Course Overview



### 1.1 Why is visual perception important?

- Computer Vision (CV) 가 왜 AI 인가
  - AI 는 사람의 지능을 컴퓨터 시스템으로 구현하는 것으로 정의할 수 있다
  - 지능에는 인지, 지각, 기억, 사고 능력이 모두 포함된다
  - CV 는 그 중에서 시각 지각 능력을 다룬다



### 1.2 What is computer vision?

- Visual World 를 Camera 등의 Sensing device 로 감지하고, GPU 와 algorithm 을 통해 해석하여,  
  Representation (재현) 하는 과정을 computer vision 이라고 한다

![image-20220316020438728](https://s2.loli.net/2022/03/16/nwmqUAlgHVNsdrc.png)



- 분석해놓은 정보를 이용하여 장면에 해당하는 이미지나, 3D 등을 재구성하면 Computer Graphics 라고 하며,    

  해당 Technique 를 Rendering 이라고 한다

![image-20220316020501631](https://s2.loli.net/2022/03/16/ejlsBpF748xtJa6.png)



- Input

  - visual data (image or video)

  

- Class of visual perception

  - Color perception
  - Motion perception
  - 3D perception
  - Semantic-level perception
  - Social perception (emotion perception)
  - Visuomotor perception, etc.



- Computer Vision 에서의 ML Vs. DL

  - Machine Learning (ML)

    - Input -> Feature Extraction -> Classification -> Output

    

  - Deep Learning (DL)

    - Input -> Feature Extraction + Classification -> Output
    - Feature Extraction 도 컴퓨터에게 맡기는 paradigm shift 발생
    - DL 을 적용하면서 더 좋은 성능을 내게되며 CV 의 급격한 발전이 시작



### 1.3 What you will learn in this course

- Fundamental image tasks
- Data augmentation and knowledge distillation
- Multi-modal learning
- Conditional generative model
- Neural network analysis by visualization





## Image Classification



### 2.1 What is classification?

- Classifier

  - A mapping f(.) that maps an image to a category level

    - 어떤 영상이 어느 category 에 속하는지 mapping 하는 함수

    

  - 입력

    - 영상

  - 출력

    - 영상이 해당하는 class

![image-20220316021734536](https://s2.loli.net/2022/03/16/VGCdZqOAhWfHEKQ.png)





### 2.2 An ideal approach for image recognition

- 모든 데이터를 갖고 있다면 (ideal 한 상황),  
  k-NN( Nearest Neighbors) 으로 해결되는 검색 문제가 된다



- 모든 데이터를 가질 수도 없으며,  
  Time complexity, Memory complexity 등의 문제로 현실적이지 않다







### 2.3 Convolutional Neural Networks (CNN)

- locally connected layer

  - 하나의 특징을 뽑기 위해서 모든 pixel 을 고려하는 FC (Fully Conneted) Layer 대신,  
    국부적인 (locally) 영역의 connection 을 고려하는 layer

  - 필요한 parameter 가 획기적으로 감소한다

    

  - 적은 parameter 로 공통된 특징을 표현할 수 있기에,  
    overfitting 도 방지할 수 있는, 영상에 적합한 네트워크 구조



- CNN 은 많은 CV tasks 의 backbone 으로 사용된다
  - classification, object detection, segmentation 등





## CNN architectures for image classification 1



### 3.1 History

![image-20220316022931357](https://s2.loli.net/2022/03/16/udOrFmLKZ2zplcN.png)





### 3.2 AlexNet

- LeNet-5

  ![image-20220316023100720](https://s2.loli.net/2022/03/16/9rEzIPlyFHCSjnO.png)

  - 매우 단순한 CNN architecture
  - Overall architecture
    - Conv - Pool - Conv - Pool - FC - FC
  - Convolution
    - 5 x 5 filters with stride 1
  - Pooling
    - 2 x 2 max pooling with stride 2

  

- AlexNet

  - AlexNet 은 LeNet-5 에 많은 영감을 받아 만들어졌다

    

  - LeNet-5 에 비해 발전된 부분

    - Bigger
      - 7 hidden layers
      - 605k neurons
      - 60 million parameters
    - Trained with ImageNet
      - large amount of data, 1.2 millions
    - Using better activation function (ReLU)
    - regularization technique
      - dropout



- Overall architecture
  - Conv - Pool - LRN - Conv - Pool - LRN -Conv -Conv - Conv - Pool - FC - FC - FC
  - 이 때 당시에는, GPU memory 가 부족하여 네트워크를 두 개로 나누어서,  
    두 개의 GPU 로 나누어 연산을 진행했다는 특징이 존재

![image-20220316023519805](https://s2.loli.net/2022/03/16/wXjPTbN43LUMrgt.png)

![image-20220316023743347](https://s2.loli.net/2022/03/16/1d6QK3npvB7qusJ.png)

- 마지막 3x3 Max Pool Layer 에서, Dense Layer 로 연결해주기위해서는,  
  Tensor 를 vector 로 변경하여 차원을 맞춰줘야한다

  - `nn.AdaptiveAvgPool2d((6, 6))` 과  
    `torch.flatten(x, 1)` 을 사용하는 방법이 있으며,  
    논문에서는 flatten 방식을 이용하였다

  

- 당시에는 GPU 를 2 개로 나누어 연산하였기에 Dense Layer 부분이 2048 로 되어있으며, (위 그림)  
  하나로 합쳐서 연산하는 경우애는 4096 이 된다 (아래 그림)



- Deprecated Components

  - LRN (Local Response Normalization)

    - 현재는 사용되지 않는 부분 (deprecated components)
    - activation map 에서 명암을 normalization 하는 역할을 한다
    - 현재는 LRN 보다는 Batch normalization 이 popular 하게 사용된다

    

  - 11 x 11 convolution filter

    - 첫번째 Convolutional Layer 의 큰 filter size
    - 현재는 사용되지 않는 부분 (deprecated components)



### 3.3 VGGNet

![image-20220316024732118](https://s2.loli.net/2022/03/16/AsMaC3fDiTrHLux.png)

- VGGNet 의 특징

  - Deeper architecture

    - 16 and 19 layers

    

  - Simple architecture

    - No local response normalization
    - Only 3x3 conv filters blocks, 2x2 max pooling

    

  - Better performance

    - Significant performance improvement over AlexNet
    - 2nd in ILSVRC14

    

  - Better generalization

    - Final features generalizing well to other tasks even without fine-tuning



- Overall architecture

  - Input

    - 224x224 RGB images
      - AlexNet 과 동일
    - Subtracting mean RGB values of training images
      - AlexNet 과 동일하게 RGB 의 평균값을 각 채널에서 빼주며 Normalization 진행

    

  - Key design choices

    - 3x3 convolution filters with stride 1

    - 2x2 max pooling operations

      

    - 적은 수의 larger conv filters 대신,  
      많은 3x3 conv layers 를 사용

      - Keeping receptive field sizes large enough
      - Deeper with more non-linearities
      - Fewer parameters

      

  - 3 fully-connected (FC) layers

    

  - Other details

    - ReLU for non-linearity
    - No local response normalization





## Further Reading



### VGGNet

- https://arxiv.org/pdf/1409.1556.pdf
