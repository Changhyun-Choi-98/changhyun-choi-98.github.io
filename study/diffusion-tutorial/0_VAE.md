---
layout: default
title: "0. VAE(Variational AutoEncoder)"
nav_exclude: true
section: study
subcategory: diffusion-tutorial
date: 2026-05-17
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

# **0. VAE(Variational AutoEncoder)**

<aside class="series-preface" markdown="1">

이 글은 앞으로 작성할 **Diffusion Model Tutorial** 시리즈의 첫 번째 준비 글이다.  

이 시리즈의 전체적인 흐름은 Lai et al.의 [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890){:target="_blank" rel="noopener noreferrer"}를 중심으로 따라갈 예정이다.

다만 DDPM(Denoising Diffusion Probabilistic Model)을 처음 이해하려면, 그 앞단에 있는 **VAE(Variational AutoEncoder)** 관점을 먼저 잡는 것이 중요하다. DDPM은 단순히 “노이즈를 조금씩 제거하는 모델”로만 볼 수도 있지만, 더 근본적으로는 VAE와 HVAE에서 이어지는 **variational latent-variable model**의 한 형태로 해석할 수 있기 때문이다. 즉, DDPM의 forward noising process는 일종의 고정된 encoder처럼 작동하고, reverse denoising process는 학습 가능한 decoder처럼 작동한다.

VAE 자체에 대한 설명은 Stanley Chan의 [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103){:target="_blank" rel="noopener noreferrer"}를 주로 참고할 예정이다. 이 자료는 VAE의 encoder, decoder, latent variable, ELBO, reparameterization trick을 비교적 직관적으로 설명하고 있어서, DDPM으로 넘어가기 전에 필요한 배경을 쌓기에 적합하다고 판단하였다.

따라서 이 글의 목표는 VAE를 독립적인 생성 모델로 깊게 완주하는 것이 아니라, 이후 DDPM을 이해하기 위해 반드시 필요한 개념들을 정리하는 것이다.

</aside>

## **1. Building Blocks of VAE**


{% include comments.html %}
