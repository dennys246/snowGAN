---
language:
- en
license: apache-2.0
tags:
- gan
- image-generation
pipeline_tag: image-generation
library_name: pytorch
---

# The Abominable SnowGAN

## Model Description
The snowGAN is a generative adversarial network built to take in magnified pictures of snowpack and train a generator and discriminator to generate and discriminate pictures of the snow respectively. The end goal is to pre-train an AI that could potentially be rebuilt to assess other things like avalanche risk or wind loading.

This is an example of the data fed into the snowGAN...
![IMG_3357](https://github.com/user-attachments/assets/23c833e4-5664-4ccf-aeb3-3defd1af1478)

This is an example of a picture generated from the snowGAN after training on ~1500 images over 50 epochs...
![IMG_3451](https://github.com/user-attachments/assets/466bdbd6-0186-488e-8f8a-fd426b7bf2d2)

This is an actively evolving project, with only 1/2 of the dataset being utilized so far. Over the Spring/Summer of 2025 data preprocessing will be finishing up and the dataset should be released publicly. At that time this project will be updated with thorough guidance for downloading the dataset, pre-trained snowGAN and other models to experiment with in your own pet or professional projects!

## Intended uses & limitations
- Intended for research and artistic generation of snow scenes.
- Not suitable for safety-critical applications.
- Model outputs may not generalize well to very different domains.

## Training data
- Dataset: Custom winter landscape dataset (~3,000 images)
- Source: Collected from open-source winter photography repositories

## Training procedure
- Architecture: Wasserstein GAN with gradient penalty
- Optimizers: Adam with learning rate 1e-4
- Number of epochs: 100

## Model Details
- **Framework**: TensorFlow
- **Generator Input**: 100D latent vector
- **Geneator Output**: 1024x1024 RGB magnified snowpack image
- **Discriminator Input**: 1024x1024 RGB magnified snowpack image
- **Discriminator Output**: 1 classification of real or fake

## Ethical considerations
- Model may produce unrealistic snow patterns — avoid misuse

## How to Use

```bash
snowgan --mode train --gen_steps 4 --disc_lr 0.0001

snowgan --mode generate --n_samples 10
``` 
