---
layout: post
title: huggingface를 이용한 한국어 BART 학습 후기
comments: true
project: true
tag:
- NLP
- Korean
- Tensorflow 2
- huggingface
---

이번에 개인적인 용도로 BART를 학습하게 되었다. 다른 사람들은 많이 쓰는 것 같은데 나는 아직 사용해본 적이 없었기 때문에 이참에 huggingface의 transformers를 써보면 좋을 것 같았다. 나는 Pretrained Model을 학습할 만한 개인 장비가 없었기 때문에 이번에도 구글의 [TPU Research Cloud](https://sites.research.google/trc/)를 지원받아서 TPU를 사용해야 했고 효율성을 위해 Tensorflow로 구현하게 되었다.

## Pretrain

BART 논문을 봤을 때는 Infilling Mask 노이즈만 사용해도 충분히 성능이 잘 나오는 걸로 보여 노이즈 함수는 infilling mask만 사용했다.

Pretraining 코드는 [**`여기`**](https://github.com/cosmoquester/transformers-bart-pretrain)에 구현해두었다.

huggingface 라이브러리를 처음 써본 결과 확실히 편하게 되어있었다. 예전에 Transformer를 TF로 쌩으로 구현해서 학습시켰을 때와 비교했을 때 model code를 구현할 필요가 없다는 점과 huggingface hub에 tokenizer와 모델을 간단히 보관하고 관리할 수 있는 것은 큰 장점인 것 같다. 또 hub에서는 간단한 inference api도 사용해볼 수 있다.

다만 huggingface tokenizer는 tensorflow-text처럼 graph에 호환되는 연산이 아니어서 pretrain할 때는 사용하지 못했다.

현재까지 학습한 모델은 mini, small, base 세 가지이고 large는 아직 학습 중이다.

### huggingface 모델 링크

[**`mini`**](https://huggingface.co/cosmoquester/bart-ko-mini)

[**`small`**](https://huggingface.co/cosmoquester/bart-ko-small)

[**`base`**](https://huggingface.co/cosmoquester/bart-ko-base)

애초에 모델을 공개할 생각으로 학습을 한 거였기 때문에 데이터셋도 공개되어 있고 누구나 구할 수 있는 것들만 사용하고자 했다.(세종 말뭉치는 이제 DVD로 받아야되는 거 같긴 한데...) 나는 대화관련 Task에 모델을 사용하려고 했기 때문에 데이터셋도 구어체에 가까운 데이터셋 위주로 찾아서 사용했다. 사용한 데이터셋은 아래와 같다.

**[모두의 말뭉치](https://corpus.korean.go.kr/)**

- 일상 대화 말뭉치 2020
- 구어 말뭉치
- 문어 말뭉치
- 신문 말뭉치

**AIhub**

- **[개방데이터 전문분야말뭉치](https://aihub.or.kr/aidata/30717)**
- **[개방데이터 한국어대화요약](https://aihub.or.kr/aidata/30714)**
- **[개방데이터 감성 대화 말뭉치](https://aihub.or.kr/aidata/7978)**
- **[개방데이터 한국어 음성](https://aihub.or.kr/aidata/105)**
- **[개방데이터 한국어 SNS](https://aihub.or.kr/aidata/30718)**

**[세종 말뭉치](https://ithub.korean.go.kr/)**

## Finetune

pretraining의 loss와 metric 만으로는 모델이 정상적으로 학습되었는지 평가하기가 어려워서 finetune에 사용할 코드도 조금 구현해보았다. 어차피 task 자체에 사용할 건 아니고 학습이 되었는지 평가하기 위한 코드라 간단한 task 몇 개를 골라서 TF를 사용해서 대충?? 짜보았다. 처음에는 BART에만 사용할 수 있게 구현했다가 다른 사람들도 쉽게 돌려볼 수 있으면 좋을 것 같아서 일반 bert나 roberta 모델 등까지 호환이 가능하게 수정했다. (나머지는 BERT나 Roberta 등도 호환되고 Chatbot과 HateSpeech는 BART만 호환된다.)

Finetune에 사용한 코드는 [**`여기`**](https://github.com/cosmoquester/transformers-bart-finetune)에 있다.

Finetune도 최대한 TPU에서 사용할 수 있도록 구현했으며 결과적으로 STS Task를 제외하고는 모두 TPU와 호환되고 링크에서 Open in colab으로 켜면 기본이 TPU이다.

STS에서 TPU가 안되는 것은 metric 중에 spearman correlation coefficient metric을 구현하는데 사용된 함수 중 일부가 TPU에서 호환이 안되서 그렇다. 해당 metric이 필요없다면 STS도 TPU에서 사용할 수 있다. 혹시라도 spearman correlation coefficient를 TPU compatible한 연산만으로 구현하는 법을 아시는 분은 알려주시면 매우 감사할 것 같습니다...(??)

아 그리고 원래 huggingface transformers에는 TFBartForSequenceClassification 모델이 없어서 이 클래스를 직접 구현해서 transformers에 넣어주는 식으로 좀 Tricky하게 사용했다. 그래야만 TFBartForSequenceClassification.from_pretrained로 hub에 올려놓은 모델을 가져올 수 있기 때문에...

## Performance

| Model | Num Parameters |
| ----- | -------------- |
| mini | 12,995,330 (13M) |
| small | 40,818,178 (41M) |
| base | 127,557,890 (128M) |
| hyunwoongko/kobart | 124,452,098 (124M) |
| klue/roberta-small | 68,090,882 (68M) |
| klue/roberta-base | 110,618,114 (111M) |

- Tensorflow에서 Sequence classifier 기준으로 구한 파라미터 수

---

| Dataset | Metric | mini | small | base | klue<br />roberta-small | klue<br />roberta-base | hyunwoongko<br />kobart |
| ----------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- |
| **KLUE NLI dev** | *Acc* | 0.5253 | 0.6390 | 0.7390 | 0.7917 | 0.8557 | 0.7527 |
| **NSMC test** | *Acc* | 0.8425 | 0.8721 | 0.8877 | 0.8977 | 0.9093 | 0.8916 |
| **QuestionPair test** | *Acc* | 0.8945 | 0.9050 | 0.9208 | 0.9367 | 0.9433 | 0.9261 |
| **KLUE TC dev** | *Acc* | 0.8047 | 0.8551 | 0.8667 | 0.8538 | 0.8636 | 0.859 |
| | *F1* | 0.7988 | 0.8515 | 0.8637 | 0.8559 | 0.8636 | 0.8557 |
| **KLUE STS dev** | *F1* | 0.7411 | 0.7406 | 0.7654 | 0.7994 | 0.8086 | 0.8025 |
| | *Pearson* | 0.7471 | 0.7593 | 0.8090 | 0.8634 | 0.8823 | 0.8190 |
| | *Spearman* | 0.7399 | 0.7551 | 0.8040 | 0.8634 | 0.8906 | 0.8134 |
| **KorSTS dev** | *F1* | 0.7725 | 0.7897 | 0.8067 | 0.8204 | 0.8249 | 0.8124 |
| | *Pearson* | 0.6503 | 0.7269 | 0.7909 | 0.8101 | 0.8200 | 0.7842 |
| | *Spearman* | 0.6191 | 0.7037 | 0.7784 | 0.8041 | 0.8152 | 0.7741 |
| **HateSpeech dev** | *Bias Acc* | 0.7537 | 0.8068 | 0.8280 | N/A | N/A | 0.8153 |
| | *Hate Acc* | 0.5605 | 0.5966 | 0.5669 | N/A | N/A | 0.6072 |

위 링크에 있는 노트북을 이용해 colab으로 측정한 성능입니다.

- 파라미터는 대부분 기본 파라미터로 돌려서 쟀고 STS task는 OOM이 잘 일어나서(TPU가 아니라 GPU라...) BatchSize를 좀 작게 하는 경우가 있습니다.
- klue/roberta-base 모델의 KLUE NLI task의 경우는 기본 파라미터로는 수렴을 하지 않아서 warm up rate과 lr를 조금 조정해서 측정했습니다.
- 제가 작성한 STS는 Bi Encoding 방식으로 보통 벤치마크에서 사용하는 Cross Encoding에 비해 일반적으로 성능이 낮게 나올 수 있습니다.
- 엄청 대충 돌린 거기 때문에 대략적인 참고용으로만 사용해주세요.

성능을 측정해보았을 때 기존 한국어 bart 모델인 hyunwoongko/kobart이나 klue에서 만든 roberta 모델에 비해 전반적으로 성능이 조금씩 낮은 편인 것 같다.
이유는 데이터셋이나 학습방법이나 파라미터 등 여러가지 요인이 있을 듯 하지만 정확히는 모르겠다.
아마 학습 방법은 논문이랑 거의 비슷할테니 버그가 있는 게 아니면 데이터셋 아니면 파라미터의 영향이 크지 않았을까?

그래도 학습 코드와 사용한 데이터셋이 모두 공개된 여러 크기의 한국어 BART 모델을 만들었다는 데에 의의를 둘 수 있지 않을까 싶다.
