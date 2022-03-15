Deeper layer 문제

→ Gradient Vanishing / exploding

→ 계산복잡도 증가

→ degradation problem (훈련 정확도의 퇴화)

![스크린샷 2022-03-15 오후 7.22.20.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2ce6ac0c-dc05-4c60-be97-9b0d4d08e6ff/스크린샷_2022-03-15_오후_7.22.20.png)

## GoogLeNet

> **②** **Inception module 구조 제안**
> 

![스크린샷 2022-03-16 오전 12.02.03.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/55de189b-f952-4e42-a7f4-c02b0ca0ec94/스크린샷_2022-03-16_오전_12.02.03.png)

하나의 레이어에 Convolution filter들을 사용해서 여러 측면으로 activation 관찰 (depthX → 수평확장)

→ 물론 한층에 여러 필터를 사용하므로 계산복잡도 증가 

|

![스크린샷 2022-03-16 오전 12.09.24.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9cd7b17b-db9a-407f-a7f8-ae4c391b55f0/스크린샷_2022-03-16_오전_12.09.24.png)

따라서, 1x1 convolution 적용 (1x1 convolution을 이용해 channel의 차원 줄임) (in ‘bottleneck layers’)

→ 계산복잡도 감소

> **위의 1x1 convolution의 구현 방식**
> 

![스크린샷 2022-03-16 오전 12.14.12.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3ab881da-a4f5-480e-8d06-f5a148f3e39a/스크린샷_2022-03-16_오전_12.14.12.png)

= 필터 개수만큼 출력 channel 생성 

→ 1x1 convolution 적용 후, 공간크기는 변하지 X. 각 pixcel에 독립적으로 channel수 바꿔줌

> **Inception module을 사용한 GooLeNet의 ‘전체구조’**
> 

![다운로드.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9c503c2d-5bfd-4fa7-8a7b-a9d64c43514a/다운로드.jpg)

③ classifier : 깊은layer로 인해 output으로부터의 back-propagation gradient가 중간에 사라지는 문제발생
                       (gradient vanishing)
→ 중간에 classifier를 둬서, 중간에 loss 계산해 back-propagation 하도록하여 아래까지 gradient를 보냄

> **③** **Auxilliary classifier에 대하여 더 자세히**
> 
- vanishing gradient problem 해결
- low layer 까지의 gradient 도달
- train에서만 사용하고, test에서는 해당 부분 제거
