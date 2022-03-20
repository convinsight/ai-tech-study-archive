# Image captioning

input image → CNN → RNN

- Encoder
  - Resnet
  - 공간 정보의 유지를 위해 model의 처음과 끝을 잘라서 사용
    - linear, pool layer
  - CNN 출력이 tensor 형태로 나오기 때문에, H와 W를 알고 있는 것이 좋음
  - (2048, 14, 14) output
- Decoder
  - RNN, Attention
  - start token
  - output과 attention 함께 다시 decoder에 input
  - 반복적으로 단어가 추출 됨

- Beam search
  - second best가 뒤의 내용에 더 알맞았을 경우
  - K
    - Best stcore K개를 뽑음
    - 각 격우에 따라 class prediction
    - 매 경우에 대해서 top K개만 남겨두며 실행
  - 중간에 남은 값과 마지막까지 뽑아낸 결과를 비교

