---
layout: post
title: "0. VAE(Variational AutoEncoder)"
nav_exclude: true
section: study
subcategory: diffusion-tutorial
date: 2026-05-30
tags:
  - Korean
  - Writing
language: ko
summary: "DDPM의 variational perspective를 이해하는 데 필요한 VAE의 핵심 개념을 정리"
math: true
comments: true
comment_id: "study-diffusion-tutorial-vae"
permalink: /study/diffusion-tutorial/vae/
---

<aside class="series-preface" markdown="1">

이 글은 앞으로 작성할 **Diffusion Model Tutorial** 시리즈의 첫 번째 준비 글이다.  

이 시리즈의 전체적인 흐름은 Lai et al.의 [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890){:target="_blank" rel="noopener noreferrer"}를 중심으로 따라갈 예정이다.

다만 DDPM(Denoising Diffusion Probabilistic Model)을 처음 이해하려면, 그 앞단에 있는 **VAE(Variational AutoEncoder)** 관점을 먼저 잡는 것이 중요하다. DDPM은 단순히 “노이즈를 조금씩 제거하는 모델”로만 볼 수도 있지만, 더 근본적으로는 VAE와 HVAE에서 이어지는 **variational latent-variable model**의 한 형태로 해석할 수 있기 때문이다. 즉, DDPM의 forward noising process는 일종의 고정된 encoder처럼 작동하고, reverse denoising process는 학습 가능한 decoder처럼 작동한다.

VAE 자체에 대한 설명은 Stanley Chan의 [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103){:target="_blank" rel="noopener noreferrer"}를 주로 참고할 예정이다. 이 자료는 VAE의 encoder, decoder, latent variable, ELBO, reparameterization trick을 비교적 직관적으로 설명하고 있어서, DDPM으로 넘어가기 전에 필요한 배경을 쌓기에 적합하다고 판단하였다.

따라서 이 글의 목표는 VAE를 독립적인 생성 모델로 깊게 완주하는 것이 아니라, 이후 DDPM을 이해하기 위해 반드시 필요한 개념들을 정리하는 것이다.

</aside>

## **0. Background**

이번, 그리고 이 category에서 앞으로 다룰 모든 게시물은 "생성 모델", 즉 generative AI에 대해 다룬다. 모델은 최근에는 neural network로 대부분 구현한다. 이것에 대해서도 다룰 것이 많으나 우선은 일반적인 "함수", 즉 input이 주어지면 그에 따른 output을 만들어주는 box라 생각해도 된다. 생성 모델은 (input이 무엇인지는 차차 다룰 것이고) output이 의미가 있는 데이터, 예를 들어 image가 되는 box이다.

## **1. Building Blocks of VAE**

![vae_overview](/study/diffusion-tutorial/images/vae_overview.png){: width="70%"}
*VAE overview*
{: .figure-caption}

VAE는 크게 2개의 모델, **encoder**와 **decoder**로 구성되어 있다. Input vector $\mathbf{x}$(image처럼 우리가 generate하고 싶은 데이터)를 encoder에 넣어주면 output으로 *latent variable $\mathbf{z}$*라는 것이 나온다(이것의 의미는 곧 설명하겠다). 이 latent variable이라는 것을 decoder의 input으로 넣어주면 decoder가 $\hat{\mathbf{x}}$을 output으로 뱉어준다. Decoder의 목적은 $\mathbf{x}$와 최대한 비슷한 것을 output으로 만들도록 하는 것이다. 우리가 생성하고 싶은 것이 image라고 할 때, 왜 굳이 image를 넣어서 image를 다시 만드려고 할까?

VAE의 두 구성요소 중 encoder는 학습 과정에서만 사용하고, 실제 "생성 모델"로 VAE를 사용할 때는 decoder만 사용한다. 즉 학습이 완료된 모델을 사용할 때는 시작이 input vector $\mathbf{x}$가 아니라 latent variable $\mathbf{z}$이다. 이 latent variable이라는 것은 무엇일까?

Latent variable은 보통 우리가 생성하고 싶은 모델보다 차원이 낮은, 그리고 생성하기 쉬운 (그래서 우리가 보기에는 의미가 없어 보이는) 변수이다. 즉, encoder는 고차원 데이터를 input으로 받아서 저차원 데이터를 output으로 뱉어주고, 그 저차원 데이터를 가지고 decoder는 고차원 데이터를 **복원**하려고 노력한다.

> **만약 encoder가 input image $\mathbf{x}$에서 의미있는 정보만 잘 압축할 수 있다면, decoder의 output이 $\mathbf{x}$와 비슷하게 자연스러운 image가 될 수 있다!**

물론 latent variable $\mathbf{z}$ 자체가 $\mathbf{x}$나 $\hat{\mathbf{x}}$보다 차원이 작기 때문에, 정보의 손실이 아예 없을수는 없다. 이것은 정보이론에서 증명되었다. 하지만 정보이론까지 갈 것도 없이 상식적으로 생각해봤을 때 실수 element의 개수가 5개인 모든 vector를 실수 element의 개수가 3개인 vector로 표현하는 것은 불가능하다. 그러니 최대한 "의미있는" 정보를 잘 압축해야 한다.

만약 encoder가 위에서 언급한 목적을 달성했고, decoder도 (뒤에서 자세히 설명할) 학습이 잘 되었다고 가정하자. 그러면 latent variable은 decoder가 생성할 image의 "씨앗"이 되어줄 것이다. 그리고 이 씨앗은 우리가 위에서 처음에 가정할 때 "생성하기 쉽다"고 가정했기 때문에 쉽게 씨앗을 만들고 이것을 decoder에 넣어준다면 결과적으로 쉽게 image를 생성한 것이 될 것이다.


{% include comments.html %}
