# Image Denoising with CNN-Based Generator

This project demonstrates a simplified approach to image denoising using a convolutional neural network (CNN). Although the generator structure resembles those used in Generative Adversarial Networks (GANs), this implementation does not include a discriminator or adversarial loss. Instead, it uses MSE to directly compare a denoised output to the original clean image.

## Key Features
- Single-image training: the model is trained on one noisy-clean image pair.
- CNN architecture with progressively deeper convolutional layers.
- Supervised loss (MSE) without adversarial training.
- Useful as a plug-and-play test bed for denoising architectures or as a stepping stone to full GAN-based approach


### TODO
- Add training loss graph
- Incorporate a discriminator for adversarial training.
- Expand to bach training with a dataset (e.g., BSDS500, DIV2K)
- Explore perceptual or adversarial loss functions for higher-quality outputs.
