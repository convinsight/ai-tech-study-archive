# Densely Connected Convolutional Networks(DenseNet)

# Abstract

### DenseNet 원리

- **shortcut connections**을 통해 네트워크를 깊게 쌓고 정확도를 높임
- 위의 원리를 토대로 **layer간 최대한 많은 연결**을 시켜서 각 layer 간의 정보 흐름을 최대한 이용하자고 제안 ⇒ DenseNet에서는 feed-forward 시, 각 layer를 **모든 다른 layer**와 연결시킴
- 기존의 convolution layer들이 L개의 layer들에 대해서 L 번의 connection ⇒  **1+2+...+L = L(L+1)/2** 번의 **direct connections** (Figure 1)
- 코드 : https://github.com/liuzhuang13/DenseNet

<details>
<summary>📎Shortcut connection (ResNet)  </summary>
<div markdown="1">       

- 레이어간의 연결이 순서대로 연속적인 것만 있는 것이 아니라, **중간을 뛰어넘어 전달하는(더하는) shortcut이 추가**된 것
- 연산은 매우 간단하고, 개념도 매우 간단하지만 이것이 gradient를 직접적으로 잘 전달하여 **gradient vanishing/exploding 문제를 해결**하는 큰 효과를 냄

</div>
</details>


### DenseNet 장점

> **기울기 소실 문제(gradient vanishing)** 완화
> 

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/63cae17e-9213-4893-9242-bb09f6bb6b27/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T233852Z&X-Amz-Expires=86400&X-Amz-Signature=60689e023cc5fbcae5fa9d1bd3cee65f777241338ae653ad11b3ebe6f2bc7f12&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- 위 그림과 같이 DenseNet 또한 ResNet 처럼 **gradient를 다양한 경로를 통해서 받을 수 있기 때문**에 학습하는 데 도움이 됩니다.

> **feature propagation 강화**
> 

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0621d563-da5a-4f2b-8073-301ba92080e9/10.gif?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T233917Z&X-Amz-Expires=86400&X-Amz-Signature=b2db2dda9091c0e73308d5eaa1e8968936f9207111de7f973a446c51bff48093&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%2210.gif%22&x-id=GetObject">

- 위 그림을 보면 **앞단에서 만들어진 feature를 그대로 뒤로 전달**을 해서 **concatenation** 하는 방법을 사용을 합니다. 따라서 **feature를 계속해서 끝단 까지 전달**하는 데 장점이 있습니다.

> **feature reuse**  → DenseNet 항목에서 설명
> 

> **parameter 개수 줄임** → DenseNet 항목에서 설명
> 

### 실험 방법

- CIFAR-10/CIFAR-100/SVHN/ImageNet dataset으로 **benchmark tasks**

# Introduction

### 배경

- CNN (Convolutional Neural Networks)는 visual object recognition에 자주 사용되나, CNN의 네트워크가 깊어질수록(= input이나 gradient가 많은 layer를 거칠수록) 네트워크 끝 부분에서는 gradient가 소실 되는(vanishing) 문제 발생  

- 이 문제를 해결하기 위해 **ResNet/Highway network/Stochastsic depth/FractalNets** 등장  

- 1️⃣ **ResNet과 2️⃣ Highway Network**는 **identity connection**(자기 자신을 다시 feed시켜주는 방식)을 사용  

- 3️⃣ **Stochastic depth**는 Resnet의 **layer를 random하게 없애주어**(dropping layer) **크기를 줄임**  

- 4️⃣ **Fractal Net**은 각기 다른 숫자의 convolutional block들로 이루어진 parallel layer들의 sequence를 여러 번 반복시켜 **short path를 유지**한 채 nominal depth(공칭두께)를 크게 하였다. (?)  

- 문제를 해결하는 핵심 : **앞 쪽의 layer와 뒤 쪽의 layer를 short path로 연결**


### DenseNet

- 최대한의 정보 흐름을 보장하기 위해서, **모든 layer**를 각각 **직접 연결**
- **L(L+1)/2**번의 **direct connections**이 이루어진다. (Figure 1)

<img width="50%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/7daf9750-96a8-42cd-a124-95ecaeb3359f/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234041Z&X-Amz-Expires=86400&X-Amz-Signature=9f7636bfd70bb5bb04af1bd77fbcee6ff054d26d893bf8e9f3100f5004fd6aab&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

> ***information preservation***
> 
- ***ResNet***은 identity transformation을 더해서(summation) **later layer로부터 early layer로의 gradient flow가 직접 연결된다는 장점**이 있지만, **identity transformation과 출력 H(x−1)이 summation됨에 따라 information flow를 방해**할 수 있다.  

    - gradient가 흐르게 된다는 점은 도움이 되지만, forward pass에서 보존되어야 하는 정보들이 **summation을 통해 변경되어 보존되지 못할 수 있다**는 의미이다. (DenseNet은 concatenation을 통해 그대로 보존)  
    
- ***DenseNet***은 feature map을 그대로 보존하면서, feature map의 일부를 layer에 **concatenation** → 네트워크에 **더해질 information**과 **보존되어야 할 information**을 분리해서 처리 → information 보존

> ***improved flow of information and gradient***
> 
- **모든 layer가 이전의 다른 layer들과 직접적으로 연결**되어 있기 때문에, loss function이나 input signal의 gradient에 직접적으로 접근 가능 + **gradient vanishing이 없어짐 →** 네트워크를 깊은 **구조로 만드는 것이 가능**

> ***regularizing effect***
> 
- 많은 connection으로 **depth가 짧아지는 효과** → **regularization 효과 (overfitting 방지)**
- 상대적으로 작은 train set을 이용하여도 **overfitting** 문제에서 자유로움

# DenseNets

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/86c94a1e-cafd-4254-a9bd-5f13c0dac315/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234113Z&X-Amz-Expires=86400&X-Amz-Signature=ce58e0f4ec02f5b520441368f21f966539a20f636c6334fe213f45a45d4a73c3&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

> ***Dense Connectivity***
> 

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/fbb06817-40b4-4b1d-b314-7ee826811ed1/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234135Z&X-Amz-Expires=86400&X-Amz-Signature=4fd13633b757b28c21aae51e4bc75d390eb171d893ac86dcc4504da82729e614&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/fcddf21e-c920-4f37-8300-780efc9abaf6/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234159Z&X-Amz-Expires=86400&X-Amz-Signature=c53d887e5453dce2c69add78f79c12e2164d2f2348d740b5bad1f3d11eda8746&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- **ResNet**은 gradient가 identity function을 통해 직접 earlier layer에서 later layer로 흐를 수 있으나, identity function과 output을 더하는(summation) 과정에서 information flow를 방해할 수 있음 → **L**번의 **connections**

<img width="30%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/2c6d5274-9d9b-4b40-922d-9a9094f167c9/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234227Z&X-Amz-Expires=86400&X-Amz-Signature=4bd52acb9d97ecbf88f5adf403f3a8ed900b5d99d350732d5f26af73180467fe&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- **DenseNet**은 summation으로 layer 사이를 연결하는 대신에, concatenation으로 layer 사이를 직접 연결 → **L(L+1)/2**번의 **connections** ⇒ **dense connectivity**라서 DenseNet(Dense Convolutional Network)으로 명명

<img width="30%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0ad0a3f2-22e5-4dd5-91ba-a1f9e48e4bf9/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234245Z&X-Amz-Expires=86400&X-Amz-Signature=200133f3d750949a0b699837a03bb500d5396434d42458e5fad79d6ad2aa6058&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">  

> ***Composite function***  
>   
- H(x)는 합성함수로, 아래의 3개 연산이 결합된 구조
    - batch normalization (BN)
    - rectified linear unit (ReLU)
    - 3 x 3 convolution (Conv)
<details>
<summary>📎Batch Normalization  </summary>
<div markdown="1">      

- `Batch` 단위로 학습을 하게 되면 발생하는 문제점 →  **Batch 단위간에 데이터 분포가 달라지는 현상**
- 이 문제를 개선하기 위해 **Batch Normalization** 개념이 적용
- **batch normalization →** 학습 과정에서 각 배치 단위 별로 데이터가 다양한 분포를 가지더라도 **각 배치별로 평균과 분산을 이용해 정규화**하는 것

</div>
</details>


> ***Pooling layers***
> 
<img width="70%" src="https://i.imgur.com/64MoJfm.png">

- feature map의 크기가 변경될 경우, **concatenation 연산**을 수행할 수 없음 (∵ 평행하게 합치는 것이 불가능) ↔ CNN은 **down-sampling**은 필수이므로, layer마다 feature map의 크기가 달라질 수 밖에 없음  

- DenseNet은 네트워크 전체를 몇 개의 dense block으로 나눠서 **같은 feature map size를 가지는 레이어들은 같은 dense block**내로 묶음  

- 위 그림에서는 총 3개의 dense block으로 나눔  

    - **같은 블럭 내의 레이어들은 전부 같은 feature map size**를 가짐 ⇒ concatenation 연산 가능  
    
    - **transition layer(**빨간 네모를 친 pooling과 convolution 부분**)** ⇒ down-sampling 가능  
    
        - Batch Normalization(BN) 
        - 1×1 convolution → feature map의 개수(= channel 개수)를 줄임
        - 2×2 average pooling → feature map의 가로/세로 크기를 줄임  
        
    - ex. dense block1에서 100x100 size의 feature map을 가지고 있었다면 dense block2에서는 50x50 size의 feature map  
    
- 위 그림에서 **가장 처음에 사용되는 convolution 연산 →** input 이미지의 사이즈를 dense block에 맞게 조절하기 위한 용도로 사용됨 → 이미지의 사이즈에 따라서 사용해도 되고 사용하지 않아도 됨
<details>
<summary>📎pooling layer  </summary>
<div markdown="1">       

- **Pooling의 필요성 ?**
- CNN에는 많은 convolution layer를 쌓기 때문에 필터의 수가 많음 → 필터가 많다 = 그만큼 feature map들이 쌓이게 된다 ⇒ CNN의 차원이 매우 크다
- 높은 차원을 다루려면 그 차원을 다룰 수 있는 많은 수의 parameter가 필요 → but, parameter가 너무 많아지면 학습 시 overfitting이 발생 →  필터에 사용된 **parameter 수를 줄여서 차원을 감소시킬 방법**이 필요 ⇒ `pooling` layer로 해결


</div>
</details>

> ***Growth rate***
> 

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0e8d61f4-e416-4808-b230-92586fde1024/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234315Z&X-Amz-Expires=86400&X-Amz-Signature=44bc4eb509aa27ba414cf5b4b0b8123282df22211089f5e61757c5e0313f6fea&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- input의 채널 개수 k_0와 이전 (l-1)개의 layer → H(x) → output으로, k feature maps (단, k_0 : input layer의 channel 개수)
    - input : k_0+k*(l-1)
    - output : k  

- **Growth rate(= hyperparameter k) →** 각 layer의 **feature map의 channel 개수**  

- 각 feature map끼리 densely connection 되는 구조이므로 자칫 **feature map의 channel 개수가 많을 경우**, 계속해서 channel-wise로 concatenate 되면서 channel이 많아질 수 있음 ⇒ DenseNet에서는 각 layer의 feature map의 channel 개수로 **작은 값**을 사용  

- **concatenation 연산**을 하기 위해서 각 layer 에서의 output 이 **똑같은 channel 개수**가 되는 것이 좋음 → 1x1 convolution으로 growth rate 조절  

- 위의 그림 1은 **k(growth rate) = 4 인 경우**를 의미  

    - 6 channel feature map인 input이 dense block의 4번의 **convolution block**을 통해 (6 + 4 + 4 + 4 + 4 = 22) 개의 channel을 갖는 feature map output으로 계산이 되는 과정
    - DenseNet의 각 dense block의 각 layer마다 feature map의 channel 개수 또한 간단한 등차수열로 나타낼 수 있음  
    
- DenseNet은 **작은 k**를 사용 → (다른 모델에 비해) **좁은 layer로 구성 ⇒ 좁은 layer로 구성해도 DenseNet이 좋은 성능을 보이는 이유?**  
 
    - Dense block내에서 각 layer들은 **모든 preceding feature map에 접근 가능** (= 네트워크의 “collective knowledge”에 접근)    
        ⇒ (생각) **preceding feature map = 네트워크의 global state**
    - **growth rate k** → 각 layer가 **global state**에 얼마나 많은 새로운 정보를 contribute할 것인지를 조절
    - ⇒ **모든 layer가 접근할 수 있는 global state로 인해 DenseNet은** 기존의 네트워크들과 같이 **layer의 feature map을 복사해서 다른 layer로 넘겨주는 등의 작업을 할 필요가 없음 (= feature reuse)**

> ***Bottleneck layers***
> 
<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/ad59f066-f5eb-466a-9108-8bbe25d09598/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234336Z&X-Amz-Expires=86400&X-Amz-Signature=8c05bd8f0062ae10e0daeb0446ba98ee049bec954ac6e5f389ba021267108ca8&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- output**의 feature map 수(= channel 개수)를 조절하는 *bottleneck layer*를 사용**
- 본 논문에서 H()에 **bottleneck layer**를 사용한 모델을 **DenseNet-B**로 표기
    - Batch Norm → ReLU → Conv (1×1) → Batch Norm → ReLU → Conv (3×3)
    - 본 논문에서, 각 1×1 Conv는 4k개의 feature map을 출력 (단, 4 * growth rate의 **4배** 라는 수치는 hyper-parameter이고 이에 대한 자세한 설명은 하고 있지 않음)
- 1x1 convolution → channel 개수 줄임 ⇒ 학습에 사용되는 3x3 convolution의 parameter 개수 줄임

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/bc0a0218-ca05-49ee-b1ba-8888d97f2db1/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234351Z&X-Amz-Expires=86400&X-Amz-Signature=b43d69b4f0b5ef9d4a806ca60bdb0b5940efa0fa6af00ff33d8e1233cdd837b4&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- **ResNet**은 `Bottleneck` 구조를 만들기 위해서
    - **1x1 convolution**으로 **dimension reduction**을 한 다음 + 다시 **1x1 convolution**을 이용하여 **expansion**
- **DenseNet**은 `Bottleneck` 구조를 만들기 위해서
    - **1x1 convolution**으로 **dimension reduction +** but, expansion은 하지 않음
    - 대신에 feature들의 `concatenation`을 이용하여 expansion 연산과 같은 효과를 만듦
        - (생각) feature들의 concatenation으로 채널 개수 expansion → ex. 6 + 4 + ... + 4

- (공통점) 3x3 convolution 전에 1x1 convolution을 거쳐서 **input feature map의 channel 개수를 줄임**
- (차이점) 다시 input feature map의 channel 개수 만큼 생성(ResNet)하는 대신 **growth rate 만큼의 feature map을 생성(DenseNet) ⇒** 이를 통해 **computational cost를 줄일 수 있음**
<details>
<summary>📎bottleneck layer  </summary>
<div markdown="1">    

- Channel 개수가 많아지는 경우, 연산에 걸리는 속도도 그만큼 증가할 수 밖에 없는데, 이때 **Channel 의 차원을 축소**하는 개념이 Bottleneck layer
- Convolution Parameters = Kernel Size x Kernel Size x Input Channel x Output Channel
- 이때, **1x1 Convolution** 을 **input 값에 Convolution** 해주면 **해당 input 의 Channel** 은 **1x1 Convolution 의 Filter 수만큼 축소**

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f0dc034c-7297-46e3-abbd-b95511819ef5/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234416Z&X-Amz-Expires=86400&X-Amz-Signature=81ffc152a64eea99d3c7c2cb6d34ebe3449ca09f793cb84101539fd712af13ad&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- 위의 그림에서와 같이 **1x1 Convolution 의 Filter 수** 만큼 **feature 의 Channel 수**가 감소
- 이렇게 줄어든 feature 를 통해 3X3, 5X5 등의 Convolution 을 위해 연산에 사용되는 **parameter를 줄여 연산의 효율성을 높임**

</div>
</details>


> ***Compression***
> 
- **Compression**은 pooling layer(Transition layer)의 **1x1 Convolution layer** 에서 **channel 개수(=  feature map의 개수)를 줄여주는 비율** (hyperparameter θ)
    - 본 논문에서는 **θ=0.5**로 설정 → transition layer를 통과하면 feature map의 개수(channel)이 **절반**으로 줄어들고, 2x2 average pooling layer를 통해 **feature map의 가로 세로 크기** 또한 **절반**으로 줄어듦
    - **θ=1**로 설정 시 → feature map의 개수를 그대로 사용

### Implementation Details

> ***CIFAR, SVHN***
> 

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/be26f054-2f4a-4564-9afa-cfa5262219e7/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234441Z&X-Amz-Expires=86400&X-Amz-Signature=5499a17e42ef001affbe7216b25a5a3d2dde9b0e76a14503ef7d2d21641f3486&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- 3 dense blocks (각 block 마다 layer 개수는 동일)
- 첫 번째 dense block 이전의 convolution는 16 output channel
- kernel size 3 x 3 + input : zero-padded by one pixel  ⇒ feature map size를 고정
- feature map size - 1st : 32 x 32 / 2nd : 16 x 16 / 3nd : 8 x 8

- DenseNet structure
    - {L=40, k=12}
    - {L=100, k=12}
    - {L=100, k=23}

- DenseNet-BC structure
    - {L=100, k=12}
    - {L=250, k=24}
    - {L=190, k=40}
    
<details>
<summary>📎zero-padded by one pixel  </summary>
<div markdown="1">       

- ex. 6x6 크기의 이미지가 있고, 3x3 필터로 convolution을 하면, output의 크기는 6-3+1로 4x4
- 1 만큼의 padding을 사용하면, 이미지의 외곽에 1 픽셀씩 더함 + zero-padding : 추가된 1 pixel에 0을 부여함
- ⇒ padding 후의 이미지 크기는 8x8이고, output의 크기는 8-3+1=6으로 6x6인 원본 이미지 크기와 같게 유지됨

</div>
</details>

> ***ImageNet***
> 

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/02784bef-e396-4d79-8606-edaade57ece0/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234503Z&X-Amz-Expires=86400&X-Amz-Signature=4ed49d2dd389494d950af3db317240f88b9cf053415378fef06c219568ae1b7a&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- 4 dense blocks
- 224 x 224 input images
- DenseNet-BC structure

> ***CIFAR, SVHN VS. ImageNet***
> 

**ImageNet**은 다른 두가지 데이터셋에 비해 이미지 사이즈가 크기 때문에 ImageNet과 나머지 두 데이터셋이 다른 architecture를 가짐

- DenseBlock 이전 Convolution 연산의 차이
- DenseBlock, Transition Layer 개수 차이
- 각 Dense Block의 layer 개수 차이
- Fully-connected layer의 output 개수(class 개수) 차이

# Experiments

### Datasets

- ***CIFAR***
    - 32 x 32 pixels
    - CIFAR-10 : 10 classes / CIFAR-100 : 100 classes
    - training set : 50,000 images / test set : 10,000 images / validations set : 5,000 training images
    - data augmentation : mirroring / shifting
    - preprocessing : normalize the data using channel means + standard deviations
- ***SVHN***
    - 32 x 32 digit images
    - training set : 73,257 images / test set : 26,032 images / validation set : 6,000 images
    - additional training set : 531,131 images
- ***ImageNet***
    - training set : 1,2 million images / validation set : 50,000 images
    - 1000 classes
    - data augmentation + 10-crop/single-crop
    - 224 x 224 images

### Training

- stochastic gradient descent (SGD)로 train
- weight decay : 10^{-4}
- Nesterov momentum : 0.9 without dampening

- ***CIFAR, SVHN***
    - batch size : 64
    - 300 or 40 epochs
    - learning rate : 0.1 → training epoch가 50%, 75%일 때 0.1배

- ***ImageNet***
    - batch size : 256
    - 90 epochs
    - learning rate : 0.1 → 30 epochs, 60 epochs마다 0.1배
    
<details>
<summary>📎momentum  </summary>
<div markdown="1">       

- parameter를 update할 떄, 현재 gradient에 과거에 누적했던 gradient를 어느정도 보정해서 과거의 방향을 반영하는 것

</div>
</details>


### Classification Results on CIFAR and SVHN

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/640bf9ee-c543-4552-82e3-065de386c555/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234529Z&X-Amz-Expires=86400&X-Amz-Signature=1d61055841b1fc4750048dc0826fceef686084350ca18848473e724d29bdb54f&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

> ***Accuracy***
> 
- **DenseNet-BC with {L=190, k=40}** → C10+, C100+에 대해 성능 좋음
- C10/C100에 대해, FractalNet with drop path-regularization 과 비교해서 error가 30% 적음
- **DenseNet-BC with {L=100, k=24}** → C10, C100, SVHN에 대해 성능 좋음
- SVHN이 비교적 쉬운 task이기 때문에, 깊은 모델은 overfitting할 수 있어서, DenseNet-BC with {L=250, k=24} 는 더 이상 성능이 개선되지 않음

> ***Capacity***
> 
- compression과 bottleneck layer가 없을 때, L과 k가 커질수록 → DenseNet의 성능이 좋아짐
    - 모델이 **더 크고(k) 더 깊어질수록(L)** 더 많고 풍부한 representation을 학습 가능
- paramter 개수가 늘어날수록 → error 줄어듦
    - Error : 5.24% → 4.10% → 3.74%
    - Number of parameters : 1.0M → 7.0M → 27.2M
    - Overfitting이나 optimization(= parameter update) difficulty가 나타나지 않음

> ***Parameter Efficiency***
> 
- DenseNet-BC with bottleneck structure + transition layer에서의 차원 축소(dimension reduction)는 parameter의 효율성을 높임
- FractalNet과 Wide ResNets는 30M parameter이고, 250-layer DenseNet은 15.3M parameter 인데, DenseNet의 성능이 더 좋음

> ***Overfitting***
> 
<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0d676db5-f6a0-4c08-9164-e213c33d8069/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234546Z&X-Amz-Expires=86400&X-Amz-Signature=dfddf33b5d765212096c072f31dae25ef39a1124a19ad650671a31710b5ecd50&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- DenseNet은 overfitting 될 가능성이 적음
- DenseNet-BC with bottleneck structure와 compression layer가 overfitting을 방지하는데 도움
- **ResNet-1001과 DenseNet-BC(L=100,k=12)의 error**를 비교 (맨 오른쪽 그래프)
    - ResNet-1001은 DenseNet-BC에 비해 **training loss는 더 낮지만**, **test error는 비슷**한 것을 알 수 있는데, 이는 **DenseNet이 ResNet보다 overfitting이 일어나는 경향이 더 적다**는 것을 보여줌
    

### Classification Results on ImageNet

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4937b36e-ed7d-4a25-8b31-e772ae221600/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234617Z&X-Amz-Expires=86400&X-Amz-Signature=ec4d71dd849c1e10623e1b7ca0369036340be7531e635d8081fd00725f55bd17&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- Table 3(왼쪽 표)은 DenseNet의 ImageNet에서의 single crop, 10-crop validation error
- Figure 3(오른쪽 그림)는 DenseNet과 ResNet의 single crop top-1 validation error를 parameter 개수와 flops를 기준으로 나타냄
    - DenseNet-201 with **20M parameters**와 101-layer ResNet with more than **40 parameter**가 비슷한 성능

# Disscussion

<img width="70%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/0d676db5-f6a0-4c08-9164-e213c33d8069/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234634Z&X-Amz-Expires=86400&X-Amz-Signature=78d2924c8edf5d204ced971014fc61d5ba786ea9a6fbfec149546ef5ef2763b5&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

> ***Model compactness***
> 
- feature map은 모든 다음 layer에 의해 접근 가능 → feature reuse → **모델이 compact 해짐**
- Figure 4의 가운데 그림에서, DenseNet-BC는 ResNet **parameter 개수의 1/3만으로도 비슷한 성과**

> ***Implicit Deep Supervision***
> 
- 각 layer마다 shorter connection을 이용하여 supervision을 손실(loss) 함수에서 얻을 수 있음 ⇒ DenseNet은 deep supervision처럼 볼 수 있음 (?)
- deeply-supervised nets (DSN)
    - 모든 hidden layer마다 classifier가 존재 → 중간의 layer마다 discriminative feature를 학습하도록 만듦(= fetaure를 분류하는 능력 학습)
- DenseNet
    - 하나의 classifier가 네트워크의 맨 위에 존재 → 2~3개의 transition layer를 통해 direct supervision을 모든 layer에 전달 ⇒ DSN과 유사
    <details>
    <summary>📎deep supervision  </summary>
    <div markdown="1">       
    
    - Deep Neural Network에서 classifier를 여러 개 두어 성능을 올리는 것

    </div>
    </details>
        

> ***Stochastic VS. deterministic connection***
> 
- **Stochastic depth**는 ResNet layer를 랜덤하게 drop하여 layer간의 direct connection을 만드는데, 이때 pooling layer는 drop되지 않아서 DenseNet connectivity pattern와 비슷함
- ⇒ DenseNet와 ResNet의 Stochastic depth은 전혀 다름에도 불구하고 **stochastic regularizer**의 효과를 낸다는 공통점
- **stochastic depth를 DenseNet의 관점에서 해석**
    - (ResNet의) **Stochastic depth**에서는, **무작위로 일부 layer를 drop**하고 이들을 둘러싸고 있던 **layer끼리 직접적으로 연결시킴 ⇒** 궁극적으론 차이가 있지만 랜덤하게 끊어진 레이어가 다른 레이어들에 연결되는 패턴이 DenseNet의 **dense connectivity 패턴과 비슷**
    <details>
    <summary>📎stochastic VS. deterministic  </summary>
    <div markdown="1">       

     - stochastic → random과 비슷하고, deterministic과는 반대이고, non-deterministic과는 종종 비슷한 개념으로 쓰임 ⇒ 인풋이 같아도 아웃풋은 다를 수 있음
    - deterministic → 같은 시퀀스 안의 다음 사건이 지금 현재 사건으로부터 결정된다는 것 ⇒ 같은 인풋을 넣으면 항상 같은 결과를 냄
    - (생각) stochastic depth → **랜덤**하게 일부 layer를 drop하고 그 layer끼리 직접 연결시킴 (stochastic) ↔ DenseNet → **DenseNet**에서는 **이미** 2개의 layer 사이가 직접적으로 연결 되어 있음 (deterministic)
   
    </div>
    </details>
    
    <details>
    <summary>📎stochastic depth regularization </summary>
    <div markdown="1">       
    
    - 네트워크에서 layer를 일정하게 drop하는 기법
    - dropout과 비슷하게 일정 확률로 특정 layer를 drop시켜 주위에 있는 다른 layer와 연결되게 만드는 것
    - Dropout을 사용하는 이유?
    - 어떤 특정한 설명변수 feature만을 과도하게 집중하여 학습하는 과적합(Overfitting)을 방지하기 위해 사용

    </div>
    </details>


> ***Feature reuse***
> 

<img width="50%" src="https://s3.us-west-2.amazonaws.com/secure.notion-static.com/84f51b46-187c-475e-966c-f16957d1a0ca/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220315%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220315T234649Z&X-Amz-Expires=86400&X-Amz-Signature=11fb101ebef6876d15527dea5d23bf0455b4965780ff93c2d2a0c6d67a00036e&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22&x-id=GetObject">

- **학습된 DenseNet의 각 layer가 실제로 preceding layer들의 feature map을 활용하는지를 실험**
    - 학습한 네트워크의 각 dense block에서, ℓ번째 convolution layer에서 s번째 layer로의 할당된 average absolute weight를 계산 (absolute는 음의 값을 갖는 weight를 고려한 것으로 보임)
- 위 그림은 dense block 내부에서 convolution layer들의 weight의 평균이 어떻게 분포되어있는지 보여줌
- **Pixel (s,ℓ)의 색깔**은 dense block 내의 conv layer s와 ℓ을 연결하는 **weight의 average L1 norm**으로 인코딩 한 것 ⇒ 각 dense block의 **weight들이 가지는 그 크기 값을 0 ~ 1 사이 범위로 normalization** 한 결과
    - 빨간색인 1에 가까울 수록 큰 값 ↔ 파란색인 0에 가까울수록 작은 값
- 실험 결과
    - 각 layer들이 동일한 block 내에 있는 preceding layer들에 weight를 분산 시킴 (∵ 각 열에서 weight가 골고루 spread되어 있음
        - ⇒ Dense block 내에서, **실제로 later layer는 early layer의 feature map을 사용하고 있음**
    - Transition layer도 preceding layer들에 weight를 분산 시킴 (∵ 가장 오른쪽 열에서 weight가 골고루 spread 되어 있음)
        - ⇒ Dense block 내에서, **1번째 layer에서 가장 마지막 layer까지 information flow가 형성되어 있음**
    - 2, 3번째 dense block은 transition layer의 output에 매우 적은 weight를 일관되게 할당 (∵ 2, 3번째 dense block의 첫번째 행에서 weight가 거의 0에 가까움)
        - ⇒ **2, 3번째 dense block의 transition layer output**은 redundant features가 많아서 매우 적은 weight를 할당(**중복된 정보들이 많아 모두 사용하지 않아도 된다는 의미**)
        - ⇒ **DenseNet-BC에서 compression θ로 이러한 redundant feature들을 compress하는 현상과 일치**
        - (생각) **Compression**은 pooling layer(Transition layer)의 **1x1 Convolution layer** 에서 **channel 개수(=  feature map의 개수)를 줄여주는 비율** (hyperparameter θ)이므로, 중복된 정보들이 transition layer에서 제거된다는 의미 →  channel 개수 감소
    - 마지막 classification layer는 전체 dense block의 weight를 사용하긴 하지만, early layer보다 later layer의 feature map을 더 많이 사용함 (∵ 3번째 dense block의 가장 마지막 열에서 weight가 아래쪽으로 치우쳐 있음)
        - ⇒ **High-level feature가 later layer에 더 많이 존재함**

    - 참고 : [DenseNet (Densely connected convolution networks) - gaussian37](https://gaussian37.github.io/dl-concept-densenet/)
    
    ![https://gaussian37.github.io/assets/img/dl/concept/densenet/24.png](https://gaussian37.github.io/assets/img/dl/concept/densenet/24.png)
    
    - 위 그림은 **각 source → target으로 propagation된 weight의 값 분포**를 나타냄
    - 세로축 `Source layer` → layer가 propagation 할 때, 그 Source에 해당하는 layer가 몇번째 layer인 지 나타냄
    - 가로축 `Target layer` → Source에서 부터 전파된 layer의 목적지가 어디인지 나타냄
    - ex. dense block 1의 세로축(5), 가로축 (8)에 교차하는 작은 사각형이 의미하는 것은 dense block 1에서 5번째 layer에서 시작하여 8번째 layer로 propagation 된 `weight`
    
    ![https://gaussian37.github.io/assets/img/dl/concept/densenet/22.png](https://gaussian37.github.io/assets/img/dl/concept/densenet/22.png)
    
    - ex. 각 dense block의 **Source가 1인 부분**들을 살펴 보면 각 Block의 **첫 layer**에서 펼쳐진 propagation에 해당 (위 그림에서 빨간색 동그라미에 해당하는 부분)
    
    ![https://gaussian37.github.io/assets/img/dl/concept/densenet/23.png](https://gaussian37.github.io/assets/img/dl/concept/densenet/23.png)
    
    - ex. 각 dense block의 **Target이 12인 부분**들을 살펴 보면 **다양한 Source에서 weight들이 모이게** 된 것을 볼 수 있음 (위 그림에서 빨간색 동그라미에 해당하는 부분)
    

### 참고

- [DenseNet (Densely Connected Convolutional Networks) (tistory.com)](https://phil-baek.tistory.com/entry/DenseNet-Densely-Connected-Convolutional-Networks)
- [DenseNet (Densely connected convolution networks) - gaussian37](https://gaussian37.github.io/dl-concept-densenet/)
- [DenseNet Tutorial [1] Paper Review & Implementation details (hoya012.github.io)](https://hoya012.github.io/blog/DenseNet-Tutorial-1/)
- [배치 정규화(Batch Normalization) - gaussian37](https://gaussian37.github.io/dl-concept-batchnorm/)
- [Global Average Pooling 이란 - gaussian37](https://gaussian37.github.io/dl-concept-global_average_pooling/)
- [[CV Study] Densely Connected Convolutional Networks (tistory.com)](https://yunmorning.tistory.com/60)
- [Dense Net(2018)논문 정리 (tistory.com)](https://aijyh0725.tistory.com/2)
- [DenseNet(Densely connected Convolutional Networks) - 2 (jayhey.github.io)](https://jayhey.github.io/deep%20learning/2017/10/15/DenseNet_2/)

- [Bottleneck layer (tistory.com)](https://realist.tistory.com/4)
- [DenseNet논문 (velog.io)](https://velog.io/@qsdcfd/DenseNet)
- [Densely Connected Convolutional Networks - 딥린이의 정리노트 (younnggsuk.github.io)](https://younnggsuk.github.io/2021/05/23/densely_connected_convolutional_networks.html)
- [Densely Connected Convolutional Networks (tistory.com)](https://eremo2002.tistory.com/116)
- [Densely connected convolutional networks (tistory.com)](https://hygenie-studynote.tistory.com/61)
- [Padding은 왜 할까? (brunch.co.kr)](https://brunch.co.kr/@coolmindory/37)
- [Densely Connected Convolutional Networks (tistory.com)](https://eremo2002.tistory.com/116)
- [[딥러닝] Drop-out(드롭아웃)은 무엇이고 왜 사용할까? (tistory.com)](https://heytech.tistory.com/127)
- [ch2 용어정리-1 stochastic이란 (velog.io)](https://velog.io/@eunice123/ch2-%EC%9A%A9%EC%96%B4%EC%A0%95%EB%A6%AC-1-stochastic%EC%9D%B4%EB%9E%80#:~:text=Stochastic%20vs.%20deterministic%20(vs.,%EB%8A%94%20%EB%B0%98%EB%8C%80%EC%9D%98%20%EA%B0%9C%EB%85%90%EC%9D%B4%EB%8B%A4.)
- [Nesterov Momentum (velog.io)](https://velog.io/@5050/Nesterov-Momentum)
- DenseNet 코드 참고 - [CV-Paper-Implementation/densenet.py at main · younnggsuk/CV-Paper-Implementation · GitHub](https://github.com/younnggsuk/CV-Paper-Implementation/blob/main/densely_connected_convolutional_networks/models/densenet.py)
