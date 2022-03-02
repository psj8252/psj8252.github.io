---
layout: post
title: "Not all memories are created equal: Learning to forget by expiring 리뷰"
excerpt: "Not all memories are created equal: Learning to forget by expiring 논문을 요약하고 간단히 리뷰했습니다."
comments: true
tag:
- NLP
- Memory
---

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css" integrity="sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X" crossorigin="anonymous">

<script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.js" integrity="sha384-g7c+Jr9ZivxKLnZTDUhnkOnsh30B4H0rpLUpJ4jAIKs4fnJI+sEnkvrMWph2EDg4" crossorigin="anonymous"></script>

<script defer src="https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/contrib/auto-render.min.js" integrity="sha384-mll67QQFJfxn0IYznZYonOWZ644AWYC+Pt2cHqMaRhXVrursRwvLnLaebdGIlYNa" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

- [Not all memories are created equal: Learning to forget by expiring](https://arxiv.org/abs/2105.06548)

## Abstract

---

- 어텐션은 장기기억이 필요한 sequence 모델링에서 좋은 성과를 보이고 있음
- 하지만 기억해야할 과거의 모든 정보의 중요도가 똑같지는 않음
- Expire-Span 이라는 중요한 정보는 유지하고 상관없는 정보는 만료(expire)시키는 방법론을 제안.
- 제안한 방법론으로 NLP나 RL Task일부에서 SOTA를 달성함.

## Introduction

---

- Transformer 아키텍처는 다양한 task에 좋은 성능을 보여줌
- 최근 연구는 어텐션을 더 긴 메모리 크기에서 효율적으로 수행하는 데 집중하고 있음
- 하지만 인간 기억의 중요한 부분에는 필요없는 정보를 잊어버리는 능력도 있음
- 메모리의 크기가 커질 수록 연관된 정보를 결정하는 것이 더 어려워짐
- 저자는 효율적으로 무엇을 잊어야할 지를 학습하는 방법에 집중하여 모델의 계산 비용을 줄이고 큰 메모리를 효과적으로 탐색하도록 만듬
- Expire-Span은 필요없는 기억을 만료시킴으로써 과거 timestep의 길이를 수만까지 확장할 수 있음
- 셀프어텐션에 매 hidden state에 expiration 값을 출력하는 간단한 predictor를 사용해 해당 정보가 얼마나 오래 보존되어야 하는지를 결정함. 이 과정은 layer간에 독립적으로 일어남.
- Expire-Span은 NLP와 RL의 삽화적 task에서 중요하고 관련없는 정보를 구별할 수 있음

## Backgroud

---

- Transformer 디코더는 feedforward와 multihead 어텐션으로 구성된 레이어들의 중첩임
- 레이어 $$l$$에서 각 timestep의 hidden state  $$\mathbf{h}_t^l \in \mathbb{R}$$ 는 key $$\mathbf{k}$$, value $$\mathbf{v}$$, query $$\mathbf{q}$$로 사상됨

$$
\mathbf{q}_t^l = W_q^l \mathbf{h}_t^l,\ \mathbf{k}_t^l = W_k^l \mathbf{h}_t^l,\ \mathbf{v}_t^l = W_v^l \mathbf{h}_t^l
$$

- (앞으로 $$l$$누락하고 싱글레이어로 설명) 이전 타임스텝의 정보는 어텐션 $$a_{ti}$$로 접근되어 $$\mathbf{o}_t$$를 생성함

$$
a_{ti} = \text{Softmax}_{i\in C_t}(\mathbf{q}_t^\top \mathbf{k}_i),\ \mathbf{o}_t = W_o \sum_{i \in C_t} a_{t,i} \mathbf{v}_i
$$

- 집합 $$C_t$$는 time $$t$$에 어떤 메모리가 액세스될 지를 보여줌
- 집합의 크기 $$\mid C_t\mid$$가 셀프 어텐션에서 시간과 공간 복잡도와 직결되는 부분이며 $$\mid C_t\mid$$를 메모리 크기라고 명명

## Method

---

- 기억 $$\mathbf{h}_i \in \mathbb{R}^d$$ 마다, 스칼라 Expire-Span $$e_i \in [0, L]$$ 을 계산함. ($$\mathbf{w} \in \mathbb{R}^d,\ b \in \mathbb{R}$$은 학습 파라미터, $$\sigma$$는 sigmoid 함수, $$L$$은 최대 span)

$$
e_i = L \sigma(\mathbf{w}^\top\mathbf{h}_i + b)
$$

- $$e_i$$는 $$\mathbf{h}_i$$가 얼마나 오래 $$C_t$$에 유지되어야할 지를 결정함
- 시간 $$t$$에서 $$\mathbf{h}$$의 남은 span은 $$r_{ti}=e_i -(t -i)$$ 로 계산하며 $$r_{ti}$$가 음수일 경우 기억 $$\mathbf{h}_i$$는 만료되어 $$C_t$$에서 제거됨
- 이 과정은 어텐션 weight $$a_{ti}$$에 바이너리 마스킹 함수 $$m_{ti} = 1_{r_{ti}>0}$$ 를 사용해서 구현할 수 있음

$$
a_{ti}^\prime = \frac{m_{ti}a_{ti}}{\sum_j m_{tj}a_{tj}},\ \mathbf{o}= \sum_i a^\prime_{ti} \mathbf{v}_i
$$

- 하지만 이렇게 이산적인 masking 함수를 사용할 경우 gradient가 전파되지 않기 때문에 저자들은 soft masking을 사용, $$R$$은 0과 0사이에서 경사도를 결정하는 hyperparameter.
    
    $$
    m_{ti} = \max(0, \min(1, 1+r_{ti}/R))
    $$
    
- 저자의 목표는 메모리 크기를 줄이는 것이기 때문에 아래와 같이 Loss에 적용함. $$\alpha>0$$는 hyperparam.

$$
\frac{1}{T} \sum_t | C_t| = R - 1 + \frac{1}{T} \sum_i \lfloor e_i \rfloor
\\
L_{total} = L_{task} + \alpha \sum_i e_i / T
$$

## Experiments and Results

---

- Expire-Span을 Transformer-XL이나 Adaptive Span 등 다른 트랜스포머 모델들과 비교해봤을 때 RL, NLP 에서 좋은 성능을 보였다.

## Conclusion

---

- Expire-Span이라는 어떤 어텐션 메커니즘에도 무엇을 잊어야할 지를 학습할 수 있는 모델을 제안
- 망각을 통해서 수만 단위까지 기억을 확장할 수 있고, LM, RL 등에서 좋은 성능을 보임
- Expire-Span은 확장성과 효율 면에서 큰 잠재력을 가지고 있다.

## Review

---

기본은 변형 트랜스포머 아키텍처를 제안하는 논문인 것 같다. 긴 sequence를 입력받을 수 있는 쪽이 많은데 이 논문은 긴 sequence에서 좀 더 필요한 정보와 아닌 정보를 좀 더 잘 구분하는 학습 방식?

- 음... 사실 그 망각이란 게 구조적으로는 기존 트랜스포머에서도 attention weight가 0에 가깝게 계산된다면  여기서 expire되는 것과 같은 기능을 할 수는 있지 않나? 왜 별도의 predictor가 필요했을까?
    - 사실 이상적으로 학습됐을 때 그냥 기존의 attention 만으로도 비슷한 기능을 할 수 있는 건 맞는 것 같다. 그런데 내가 생각하기에 핵심적인 부분은 memory의 크기를 loss에 추가했다는 점? 그래서 가능한 메모리의 크기를 작게 유지하면서도 task는 풀 수 있어야 하니까 좀 더 중요한 정보와 그렇지 못한 정보를 구별하는 능력을 배우게 된 것 같다. 그래서 기존의 Transformer 구조에서도 attention weight의 총합을 제한하거나 하는 식의 loss를 학습에 반영한다면 비슷한 결과를 얻을 수 있지 않을까 추측.
- 왜 decoder에만 적용할 수 있나?
    - 논문을 보면 여기서 transformer의 입력을 time series로 가정하고 있다. 그리고 시계열 상 뒤쪽의 데이터가 앞쪽의 데이터에 대해 attention하는 것을 줄여준다. 아무래도 그래서 decoder 레이어에 적용된다고 짤막하게 써있었던 것 같다. 시계열을 가정했는데 인코더는 bi-directional 하니까 앞뒤 순서개념이 없고 expire란 개념이 존재할 수 없는 것. 이 부분의 제약이 좀 아쉬운 것 같다.
