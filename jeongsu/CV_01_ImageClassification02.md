Deeper layer 문제

→ Gradient Vanishing / exploding

→ 계산복잡도 증가

→ degradation problem (훈련 정확도의 퇴화)

![CV_01_ImageClassification02_01.png](/assets/CV_01_ImageClassification02_01.png)

## GoogLeNet

> **②** **Inception module 구조 제안**
> 

![CV_01_ImageClassification02_02.png](/assets/CV_01_ImageClassification02_02.png)

하나의 레이어에 Convolution filter들을 사용해서 여러 측면으로 activation 관찰 (depthX → 수평확장)

→ 물론 한층에 여러 필터를 사용하므로 계산복잡도 증가 

|

![CV_01_ImageClassification02_03.png](/assets/CV_01_ImageClassification02_03.png)

따라서, 1x1 convolution 적용 (1x1 convolution을 이용해 channel의 차원 줄임) (in ‘bottleneck layers’)

→ 계산복잡도 감소

> **위의 1x1 convolution의 구현 방식**
> 
![CV_01_ImageClassification02_04.png](/assets/CV_01_ImageClassification02_04.png)

= 필터 개수만큼 출력 channel 생성 

→ 1x1 convolution 적용 후, 공간크기는 변하지 X. 각 pixcel에 독립적으로 channel수 바꿔줌

> **Inception module을 사용한 GooLeNet의 ‘전체구조’**
> 

![CV_01_ImageClassification02_05.png](/assets/CV_01_ImageClassification02_05.png)

③ classifier : 깊은layer로 인해 output으로부터의 back-propagation gradient가 중간에 사라지는 문제발생
                       (gradient vanishing)
→ 중간에 classifier를 둬서, 중간에 loss 계산해 back-propagation 하도록하여 아래까지 gradient를 보냄

> **③** **Auxilliary classifier에 대하여 더 자세히**
> 
- vanishing gradient problem 해결
- low layer 까지의 gradient 도달
- train에서만 사용하고, test에서는 해당 부분 제거
    
![CV_01_ImageClassification02_06.png](/assets/CV_01_ImageClassification02_06.png)
