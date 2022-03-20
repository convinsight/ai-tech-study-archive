# Multi-modal Learning

한 타입의 데이터가 아닌 텍스트, 사운드 등 다른 데이터를 가지고 있는 데이터를 가지고 학습을 하는 경우

#### 1.  Overview of multi-modal learning

- Multi-modal vs Unimodal
  - 다양한 형태의 데이터를 사용해서 학습하는 것을 의미
  - 사람의 기준으로 다양한 감각을 함께 쓰는 것을 Multi-modal, 하나의 감각만 쓰는 것을 Unimodal
  - 입력되는 데이터의 형태 또한 고려할 수 있음
  
- Challenge
  - 데이터마다 입력되는 데이터의 차이가 큼
    - 소리는 wave, 텍스트는 정의 자체가 어려운 상태 등
  - Unbalance
    - 1:N matching
  - model
    - 특정 modality에 편향되는 경향성 발생
  
  이러한 어려움이 존재하지만, 여전히 multi-modal learning은 중요한 요소로 꼽힘
  
- method

  - 두 데이터를 하나의 공간에서 matching하여 training
  - 하나의 데이터를 다른 modality로 translation
  - 다른 modality가 하나의 modality를 참조하여(referencing) 출력

#### 2. Multi-modal tasks (1) - Visual data & Text

1.  Text embedding
   - Character는 machine learning에 사용하기 어려움
     - 일반적으로 word 단위로 사용
   - map to dense vectors(embedding vector)
     - 단어마다 정보에 대한 해석 가능한 정도를 값으로 보유
   - Text embedding의 merging 이 generalization power를 가짐
   - word2vec: skip-gram model
     - W와 W'으로 학습
     - one hot vector와 W가 곱해지면 하나의 row가 선택됨
       - 각 row = word embedding vector
     - 하나의 word에 대해 이웃한 N개의 word를 예측하는 방향으로 학습
       - 단어간의 관계성을 이해
2. Joint embedding
   - Matching을 하기 위한 공통의 embedding vector를 얻는 절차
   - Image tagging
     - 주어진 이미지에 태그를 만들 수 있고, 태그로부터 이미지를 찾을 수 있음
     - pre-trained unimodal models를 하나로 합침
       - d-dimesional text data vector와 d-dimesional Gaussian model을 추출해 embedding
     - Metric learning
       - embedding space상에서의 거리를 조절
       - push & pull
2. Cross modal translation
   - Image Captioning
     - 이미지와 문장 간의 상호 전환
     - image: CNN, RNN: RNN
     - Show, attend, and tell
       1. input image
       2. CNN을 통해 14x14 공간 정보를 유지한 feature map 형태로 추출
       3. RNN에서 feature map을 반복적으로 referencing하며 단어 생성
     - Attention
       - 사람이 얼굴을 볼 때 주의를 기울이는 특징적인 부분을 참고한 기법
       - Inference 단계에서 어느 부분을 주목할 지 제공
       - RNN 단계에서 attention s는 feature map과 함께 z 를 생성하고 y와 함께 h를 만듦
       - h는 다시 s를 생성하고, feture map을 다시 참고해 z를 생성하는 과정을 반복함
     - Text to Image by Generative model
       - text 전체를 fixed dimensional vector로 만드는 network 통과
       - Gaussian random code(다양한 output이 나오도록, 일종의 cGAN)
       - decoder를 통해 generation
4.  Cross modal reasoning
   - Visual question answering
     - 영상과 질문이 주어지면 답을 도출하는 task
     - Image stream
       - pre-trained된 NN을 통해 fixed dimensional vector로 출력
     - Question stream
       - text sequence를 RNN을 통해 fixed dimensional vector로 출력 
     - 두 vector를 Point-wise multiplicatio해서 두 개의 embedding feature가 interaction할 수 있도록 만듦
       - 일종의 jointembedding
       - end to end

#### 3. Multi-modal tasks (2) - Visual data & Audio

1. Sound representation
   - 기본적으로 1D signal
     - 시간축에 대한 wave form
   - NN이나 ML에서 쓰이는 형태는 Spectrogram 등의 Acoustic feature의 형태
   - Wave → Spectrogram
     - Fourier transform
       - 시간축에 따른 wave form전체를 주파수 축으로 옮김
       - 들어온 주파수의 삼각함수가 어느 정도 성분으로 들오왔는 지 분해하는 툴
     - Short-time Fourier transform(STFT)
       - 시간에 따른 변화를 파악하기 위해 나누어 수행하는 방법
     - Hamming window
       - 가운데 부분에 초점을 맞출 수 있도록 windowing
       - 적당한 간격을 두고 window를 적당히 overlap하며 추출
     - Spectrogram
       - 시간에 따라 주파수 성분의 변화를 표현
       - Melspectrogram, MFCC..
2. Joint embedding
   - sound tagging
   - SoundNet
     - 오디오의 표현을 어떤 식으로 학습할 것인가
     - pre-trained된 visual recognition network 사용
     - unlabeled video을 입력하여 object distripution과 scene distribution 추출
     - raw waveform형태로 audio 추출하여 CNN구조로 입력해 2개의 head로 분리
       - 하나는 scene distribution을 따라해 place recognition 하도록 학습
       - 하나는 object distribution을 따라하도록 학습
       - tranfer learning?
       - 이 당시에는 spectrogram에 대해 딱히 고려하지 않음
     - target task에 대해 pool5 feature를 추출하여 사용가능
       - sound 학습 layer
       - classifier를 올려 학습
       - head보다 pool5가 상대적으로 generalizable 할 것으로 추측
2.  Cross modal translation
   - Speech2Face
     - 음성으로부터 얼굴을 상상해내는 task
     - Module networks
       - 담당 부분이 잘 학습된 모델을 활용하는 것
     - 인터뷰 영상으로 학습
     - face decoder는 face feature에 호환되도록 설계돼 바로 reconstruction
     - self supervise방법으로 annotation 필요 없음
   - Image-to-speech
     - 14x14 feature map
     - show, attend, and tell에서 사용한 구조를 사용해 sub-word 형태로 만듦
     - Unit-to-Speech Model을 Tacotron2를 활용해 학습
       - Tacotron2: TTS Text to Speach
       - Sub-word 단위로 학습
4. Cross modal reasoning
   - sound source localization
     - sound와 image가 입력됐을 때 소리가 나는 위치를 localization
     - VisualNet(feature vector) + AudioNet(fixed dimensional vector) → AttentionNet
       - 내적을 통해 관계성을 파악
       - fully supervised learning
     - Unsupervised version
       - localization score와 VisualNet feature map을 weighted sum pooling을 통해
         image captioning과 유사하게 Attended Visual feature 추출
       - 추출한 Attended Visual feature와 AudioNet에서 나온 sound feature를 metric learning
         - 왜 두 벡터를 metric learning?
           - VisualNet에서 나온 feature가 Sound에서 발생한 feature가 닮았으면 좋겠다는 것
     - supervised version과 함께 semi-supervised network 구성
   - Audio-visual fusion
     - 두 개의 분명하게 분리된 비디오를 따로 및 합성한 spectrogram으로 훈련
