---
layout: post
title: Tensorflow2 기반 ASR 모델, 학습 코드 구현
comments: true
project: true
excerpt: Tensorflow2을 이용해 LAS, DeepSpeech2 구조의 ASR 모델들을 구현하였습니다. 또한 학습/평가/추론 코드도 함께 구현하였고, 구현한 모델을 GPU에서 학습하여 간단히 실험하고 결과를 첨부하였습니다.
tag:
- Automatic Speech Recognition
- Tensorflow 2
- Develop
---

# ASR 모델 및 학습/평가 코드 구현

이번에 모두의 연구소를 처음했는데 ASR Lab에서 활동하면서 음성인식 모델을 구현해보게 되었습니다.  
사실 음성분야는 이번에 아예 처음이었는데 그냥 평소하던 NLP말고 다른 분야는 어떤지 궁금해서 신청하였습니다.  
덕분에 음성인식도 기본적으로 어떻게 할 수 있는지는 알 수 있었던 거 같습니다.

학습코드는 Tensorflow 2를 사용해 구현하였습니다.

상세한 코드는 [이곳](https://github.com/cosmoquester/speech-recognition)에서 확인하실 수 있습니다.

이 코드를 이용해 [Libri Speech](https://www.openslr.org/12)로 LAS모델을 학습시켜보았습니다.

학습은 모두 TPU에서도 가능은 하지만 LAS의 경우 RNN구조에서 이슈 때문에 메모리가 매우 많이 필요하게 되어 Titan RTX GPU 2대를 사용해 학습을 진행했습니다.

이번에 구현을 하면서는 특히 Tensorflow 이슈를 많이 찾았던 거 같습니다.

- [Error occured Bidirectional layer with TPU #48880](https://github.com/tensorflow/tensorflow/issues/48880)

- [LSTM and GRU on cudnn with mask puts different output from CPU or Non-cudnn kernel #49241](https://github.com/tensorflow/tensorflow/issues/49241)

- [tf.range not works with fixed input parameter on TPU #49469](https://github.com/tensorflow/tensorflow/issues/49469)

세 개 다 이슈를 남겼는데 몇 달이 지났는데 고쳐지지 않는군요... 

사실 뭐 되게 특이한 기능은 쓴 건 아니라고 생각하는데 Tensorflow 사용자가 Pytorch 사용자에 비해 숫자도 훨씬 적어서 그런지 비교적 단순한 부분도 문제가 좀 있는 것 같습니다.  
여러분들도 혹시나 Tensorflow를 쓸 때 RNN 계열을 써서 단순 Sequence Classification 이상의 것을 하려면 주의하시기 바랍니다.

실제로 학습을 제대로 테스트해본 건 LAS 모델 뿐이며 Libri Speech로 학습한 모델입니다.  
DeepSpeech나 한국어 데이터셋을 이용한 학습도 해보고 싶었지만 학습 기간이 너무 오래 걸리기도 하고 다른 할 것들도 많다보니 직접 결과를 확인한 건 LAS small Libri Speech 모델 뿐입니다.

혹시라도 학습한 모델을 돌려보고 싶으신 분은 [**여기**](https://github.com/cosmoquester/speech-recognition/releases/tag/v0.0.1)에서 다운받을 수 있습니다.

# 결과

Evaluate 결과는 아래와 같습니다.

| | LibriSpeech dev-clean | LibriSpeech dev-other |
| --- | --- | --- |
| WER (Word Error Rate) | 9.35% | 24.53% |
| CER (Character Error Rate) | 4.24% | 13.29% |

사실 SpecAugment같은 것도 구현해두었는데 실험해보지 못한 건 좀 아쉬운 거 같습니다.  
그랬다면 dev-other에서 이렇게 성능이 크게 떨어지지는 않았을 거 같은데 말이죠.

몇몇 예시를 보여드리면...

| Prediction | Target |
| --- | --- |
| THE WANDERING SINNER | THE WANDERING SINGER |
| I'LL PLAY FOR YOU NOW NEED THE APPLE BOW AND YOU SHALL DREAM ON THE LAWN SO SHADY LADY LADY MY FAIR LADY OH MY APPLE GOLD LADY | I'LL PLAY FOR YOU NOW NEATH THE APPLE BOUGH AND YOU SHALL DREAM ON THE LAWN SO SHADY LADY LADY MY FAIR LADY O MY APPLE GOLD LADY |
| THE LADIES | THE LADIES |
| NOW HE MAY PLAY A SERENE SINGER A DREAM OF KNIGHT FOR AN APPLE GOLD PLATY FOR THE FRUIT IS NOW ON THE APPLE BOUGH AND THE MOON IS UP AND THE LONG IS SHADY SINGER SINGER WANDERING SINGER O MY HONEY SWEET SINGER | NOW YOU MAY PLAY A SERENA SINGER A DREAM OF NIGHT FOR AN APPLE GOLD LADY FOR THE FRUIT IS NOW ON THE APPLE BOUGH AND THE MOON IS UP AND THE LAWN IS SHADY SINGER SINGER WANDERING SINGER O MY HONEY SWEET SINGER |
| ONCE MORE THE SINGULAR PLAYS IN THE LADIES DANCE BUT ONE BY ONE THEY FALL ASLEEP TO THE DROWSY MUSIC AND THEN THE SINGER STEPS INTO THE RING AND UNLOCKS THE TOWER AND KISSES THE UPPEREST DAUGHTER | ONCE MORE THE SINGER PLAYS AND THE LADIES DANCE BUT ONE BY ONE THEY FALL ASLEEP TO THE DROWSY MUSIC AND THEN THE SINGER STEPS INTO THE RING AND UNLOCKS THE TOWER AND KISSES THE EMPEROR'S DAUGHTER |
| I DON'T KNOW WHAT BECOMES OF THE LADIES | I DON'T KNOW WHAT BECOMES OF THE LADIES |
| BEDTIME CHILDREN | BED TIME CHILDREN |
| YOU SEE THE TREATMENT IS A TRANQUIL FANCIFUL | YOU SEE THE TREATMENT IS A TRIFLE FANCIFUL |
| HOW WE MUST SIMPLIFY | HOW WE MUST SIMPLIFY |

전반적으로 비슷하면서도 조금씩 헷갈려하는 모습?

github의 코드는 라이센스에 따라 자유롭게 사용해주시면 되고 혹시나 사용하다가 문제를 발견하면 이슈를 남겨주시면 감사할 것 같습니다. 😊
