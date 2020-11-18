---
layout: post
title: 한국어 띄어쓰기 교정 모델 개발 (Quickspacer)
comments: false
project: true
tag:
- Machine Learning
- NLP
- Korean
---

Convolution을 이용한 간단한 한국어 띄어쓰기 교정 모델입니다. 속도를 우선으로 고려하였습니다.
아래에서 바로 사용해보실 수 있습니다. 이 데모의 모델들은 [모두의 말뭉치](https://corpus.korean.go.kr) **국립국어원 문어 말뭉치(버전 1.0)** 데이터를 이용하여 학습한 모델입니다.

파이썬으로 설치해서 사용하고 싶으시거나 더 자세한 사항은 [Quickspacer 레포지토리](https://github.com/psj8252/quickspacer)를 참고해주세요.

Level은 1,2,3 세 종류가 있으며 Level이 높을수록 일반적인 띄어쓰기 성능이 높지만 더 추론이 오래걸립니다. 버튼을 눌러보면 시간이 왜 이리 오래 걸리지 싶으실 수 있는데 웹에서 Javascript로 돌리는 것을 감안해주시면 좋겠습니다. (파이썬으로 설치해서 사용하거나 Tensorflow Serving으로 serving하면 당연히 훨씬 빠릅니다.)

본 모델은 띄어쓰기를 추가하는 기능만 있기 때문에 띄어쓰기를 없앨 필요가 있다면 그냥 띄어쓰기를 모두 없앤 상태에서 띄어쓰기 하시면 됩니다. `Remove Spaces`버튼으로 띄어쓰기를 모두 없앨 수 있습니다.

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.7.0/dist/tf.min.js"></script>
<script src="../assets/post_files/quickspacer/spacer.js"></script>

# Demo
<textarea style="width:100%" rows="8" id="original-text"></textarea>
<button class="btn btn-primary btn-lg" id="remove-space-button" style="color:red">Remove Spaces</button>
<button class="btn btn-primary btn-lg" id="submit-button1">Level1</button>
<button class="btn btn-primary btn-lg" id="submit-button2">Level2</button>
<button class="btn btn-primary btn-lg" id="submit-button3">Level3</button>
<textarea readonly="" style="width:100%" rows="8" id="spaced-text"></textarea>
<label id="elaspsed-time">Elaspsed Time: 0</label>
