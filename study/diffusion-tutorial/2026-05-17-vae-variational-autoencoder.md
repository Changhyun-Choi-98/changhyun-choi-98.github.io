---
layout: default
title: "1. VAE(Variational AutoEncoder)"
nav_exclude: true
section: study
subcategory: diffusion-tutorial
date: 2026-05-17
tags:
  - Korean
language: ko
summary: "Diffusion 모델을 이해하기 위한 VAE의 핵심 개념을 정리합니다."
math: true
comments: true
comment_id: "study-diffusion-tutorial-2026-05-17-vae-variational-autoencoder"
permalink: /study/diffusion-tutorial/vae-variational-autoencoder/
---

# 1. VAE(Variational AutoEncoder)

## 목표

VAE는 입력 데이터를 낮은 차원의 latent variable로 압축한 뒤, 그 latent representation에서 다시 원본과 비슷한 데이터를 복원하도록 학습하는 생성 모델입니다. Diffusion Tutorial에서는 VAE를 먼저 다루면서 latent space, encoder, decoder, reconstruction loss, KL divergence의 역할을 정리합니다.

## 핵심 아이디어

- Encoder는 입력 $x$를 latent distribution의 파라미터인 $\mu$와 $\sigma$로 변환합니다.
- Reparameterization trick은 $z = \mu + \sigma \odot \epsilon$ 형태로 샘플링을 미분 가능하게 만듭니다.
- Decoder는 샘플링된 latent variable $z$에서 입력을 복원합니다.
- Loss는 reconstruction loss와 KL divergence를 함께 사용합니다.

## Diffusion과의 연결

Stable Diffusion 계열 모델은 이미지를 pixel space가 아니라 VAE가 만든 latent space에서 다루는 방식으로 계산량을 줄입니다. 따라서 VAE를 이해하면 latent diffusion model에서 이미지가 어떻게 압축되고 복원되는지, 그리고 diffusion process가 어떤 공간에서 수행되는지 더 명확하게 볼 수 있습니다.

{% include comments.html %}
