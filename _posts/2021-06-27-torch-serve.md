---
layout: post
title: TorchServe 사용법과 후기
comments: true
excerpt: TorchServe의 간략한 사용법과 전반적인 사용후기를 정리하였습니다.
tag:
- Model Serving
---

이번에 회사에서 Pytorch로 개발된 모델을 서빙할 필요가 있었다. 그런데 급한 건 아니었고 간단히 만들면 되서 전부터 생각하고 있던 TorchServe를 써보기로 했다.

# TorchServe 적용 순서

---

Mar파일을 만드는 부분이 대부분이며 그 이후에 약간의 configuration이 필요하다.

## 1. Eager Mode or Torchscript

MAR파일을 만들 때 모델을 Eager 모드로 사용할 수도 있고 torchscript를 사용할 수도 있다. 하지만 Eager모드로 사용하려면 모델 클래스 정의 파일을 만든 다음 MAR파일 생성 시 넣어줘야하고 Eager excution이기 때문에 나는 넣을 코드를 줄이고 성능 상의 이점을 살리고자 jit 컴파일을 통해 torchscript로 만들어서 사용했다.

## 2. Torchscript using Tracing or Script

Torchscript는 tracing이나 scripting으로 만들 수 있다. tracing은 tensorflow처럼 샘플 입력으로부터 모델을 거치는 과정을 tracing하여 torchscript로 만드는 방법이며, script은 코드 자체를 보고 torchscript로 변환하는 방법으로 보인다. 그런데 scripting 방식은 script 자체에 *을 사용한다거나 그런 모호한 부분이 있으면 사용할 수 없다. 우리 모델은 script 상에 *과 같은 ambiguous함이 있어서 불가능했기 때문에 tracing 방법을 사용해 Torchscript로 변환했다.

## 3. MAR 파일 만들기

MAR 파일을 만들 때 handler라는 게 필요하다. handler는 request로부터 데이터를 받고 "전처리 → 모델 추론 → 결과 후처리" 까지의 과정을 정의한다.

사실 상 handler가 Request를 받고 Response를 보내는 Server 파트 이외의 로직 전부를 담당한다고 볼 수 있다.

[이 문서](https://github.com/pytorch/serve/blob/master/docs/default_handlers.md)에 Pytorch에서 제공하는 기본 핸들러가 소개되어있다.

제공하는 기본 핸들러는 아래 4종 뿐이다.

- image_classifier
- image_segmenter
- object_detector
- text_classifier

### 기본 핸들러를 사용하는 경우

기본핸들러를 사용할 수 있다면 [여기](https://github.com/pytorch/serve/tree/master/model-archiver#creating-a-model-archive)의 예제처럼 `--handler` 옵션으로 해당 핸들러를 지정하고 `index_to_name` 과 같은 extra file이 필요한 경우 함께 넣어주면 MAR파일로 만들어진다.

그런데 [여기](https://github.com/pytorch/serve/blob/master/ts/torch_handler/text_classifier.py)를 보면 알겠지만 text_classifier의 경우 torchtext vocab(심지어 0.4.0버전...)를 넣어주는 형태만 쓸 수 있는 등... 제공하는 기본 핸들러도 유연하지 못하고 legacy에 강하게 bind되어 있다.

### 기본 핸들러를 사용하지 못하는 경우

사실 대부분의 일반적인 경우라면 기본핸들러를 사용하지 못할 것으로 예상된다.

이 경우에는 [예제](https://github.com/pytorch/serve/blob/master/docs/custom_service.md)처럼 custom handler를 직접 구현하여 사용해야 한다.

기본적으로 handler는

- initialize(self, context)
- handle(self, data, context)

이 두 함수를 구현하면 된다. [BaseHandler](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py) 구현 참고. Custom Handler구현에 대해서 그리 많은 고민이 들어간 것 같지는 않다. 가장 중요한 부분인 context와 data의 구조에 대한 설명도 너무 부족한 느낌.

그나마 다행인 점은 나중에 serving할 때 이 핸들러를 파이썬 그대로 실행시키기 때문에 디버깅이 어렵지는 않다는 점?

**외부 파일 넣는 법**

```python
properties = ctx.system_properties
model_dir = properties.get("model_dir")
source_vocab_path = os.path.join(model_dir, "source_vocab.pt")
```

text_classifier 핸들러를 보면 위와 같은 방식으로 context로부터 파일이 들어있는 directory를 가져와서 hard한 파일명을 통해 vocab을 불러오게 되어있다.

```python
source_vocab = self.manifest['model']['sourceVocab'] if 'sourceVocab' in self.manifest['model'] else None
if source_vocab:
    # Backward compatibility
    self.source_vocab = torch.load(source_vocab)
```

분명 윗 부분 코드를 보면 manifest에서 sourceVocab를 읽어올 수도 있는 것 같지만

[Text Classification 예제](https://github.com/pytorch/serve/tree/master/examples/text_classification)를 보면 그냥 `-extra-files` 인자로 해당 `source_vocab.pt` 라는 고정된 파일명을 넘겨주고 있고, manifest에 대해서는 문서 어디에도 별도의 언급이 없다. 우리도 외부 파일이 필요해서 같은 방식으로 넣었다.

**외부 파이썬 패키지 사용하는 법**

전처리 등에 pytorch 함수 뿐만이 아니라 외부 함수를 사용할 필요가 있다.

[이 문서](https://github.com/pytorch/serve/tree/master/model-archiver#model-specific-custom-python-requirements)를 보면 `--requirements-file` 인자를 통해 설치가 필요한 외부 파이썬 패키지를 사용할 수 있음. 그런데 requirements가 MAR파일에 포함되는 것이 아니라 그 텍스트파일이 그대로 MAR에 들어가고 serving을 시작할 때 설치하는 것. 또한 서버 config 상 기본적으로 외부 모듈 설치 옵션이 꺼져있기 때문에 config 파일에 직접 `install_py_dep_per_model=true` 로 설정해줘야함. (아니 저 파일을 넘겨줬으면 당연히 깔아줘야 되는 거 아닌가?)

그런데 어이가 없었던 문제가 있었는데 sentencepiece를 사용하고 싶은데  pytorch/torchserve 도커 이미지에 설치된 pip가 9.0.1이라서 upgrade를 해주지 않으면 설치가 되지 않았다. 결국 새 도커이미지를 만들지 않고 실행하기 위해서는 실행 command에 pip upgrade를 추가해줘야 했다........

이 과정을 거치고 나면 MAR파일이 만들어진다.

## 4. Serving 시작하기

기본적으로는 아래와 같은 명령어로 실행할 수 있다.

```bash
torchserve --start --models all --ts-config config.properties --model-store model-store
```

- model-store는 mar파일들이 들어있는 디렉토리
- models는 로드할 모델을 뜻함. 디렉토리에 있거나 s3, http등 외부에 있을 수 있다.
- ts-config는 torch serve의 config 파일

Serving 세부사항은 [여기](https://github.com/pytorch/serve/blob/master/docs/server.md) 참조. config파일 세부사항은 [여기](https://github.com/pytorch/serve/blob/master/docs/configuration.md) 참조.

### Docker로 실행하기

[여기](https://github.com/pytorch/serve/blob/master/docker/README.md)를 보면 세부사항이 나와있으며 기본적으로 torchserve 커맨드가 실행되는 것이라 필요한 모델이나 config를 마운트하고 실행 인자를 줘서 실행하면 된다.

# 종합의견 및 비교

**장점**

TorchServe는 서버만 java일 뿐 기본적으로 파이썬으로 전처리부터 추론 로직, 후처리까지 다 실행시키기 때문에 모든 operation이 tensorflow op으로만 짜야하는 tensorflow serving보다 훨씬 유연한 로직을 짤 수 있다. 

공통된 Docker 이미지하나에 모델 파일(+경우에 따라 config파일)만 바꿔서 모델을 Serving할 수 있다. 모델 서빙로직을 어쨌든 MAR파일 하나로 만들 수 있다는 점. 웹이나 s3 같은 곳에 있는 모델 파일을 바로 불러올 수 있다는 점.

Management API를 통해 실시간으로 모델을 추가하거나 정보를 확인할 수 있는 기능을 기본으로 제공한다. (해보지는 않았는데 문서보면 그렇게 써있다.)

**단점**

우선 TF Serving과 비교했을 때 너무너무 조잡한 거 같다는 인상을 받았다.

전처리가 많이 필요한 텍스트라서 더 그럴 수도 있을 거라 생각하지만 tokenize부터 detokenize까지 saved_model하나로 만들어서 바로 serving할 수 있는 TF serving과는 다르게 결국 handler파일 안에서 전처리, 후처리 로직을 하나하나 다 짜서 넣어야 한다. torch에는 string 텐서가 없고 그러다보니 torchscript만으로 serving할 수가 없어서 그런 것 같다.

그리고 문서부터 docker 이미지까지 중간중간 디테일이 너무 조잡하게 처리되어 있는 것 같다.

쌩으로 서빙했을 때 짜야할 같은 코드를 TorchServe에 맞춰서 더 복잡하고 보기 힘들게 짜야한다...
