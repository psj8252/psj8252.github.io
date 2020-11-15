---
layout: post
title: 한국어 띄어쓰기 교정 모델 개발
comments: false
project: true
tag:
- Machine Learning
- NLP
- Korean
---

CNN을 이용한 한국어 띄어쓰기 교정 모델입니다. 속도를 가장 우선으로 고려하였습니다.
아래에서 바로 사용해보실 수 있습니다. 현재 모델은 임시 모델입니다.

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.7.0/dist/tf.min.js"></script>
<script src="../assets/post_files/kor-spacer-model/spacer.js"></script>

# Demo
<textarea style="width:100%" rows="8" id="original-text"></textarea>
<button class="btn btn-primary btn-lg" id="submit-button">띄어쓰기</button>
<textarea readonly="" style="width:100%" rows="8" id="spaced-text"></textarea>

