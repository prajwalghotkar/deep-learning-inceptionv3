# deep-learning-inceptionv3
https://inceptionv3imageclassification.netlify.app/

https://github.com/prajwalghotkar/CNN

# InceptionV3 Image Classification System — From Scratch to Deployment

## Project Overview

This project presents a complete, end-to-end implementation of the InceptionV3 deep learning architecture, built from scratch using Python, TensorFlow, and Keras. The goal was to go beyond simply using a pre-built model — instead, every layer, every block, and every connection was manually designed and coded, replicating the original architecture described in Google's landmark 2015 research paper.

The project covers the full pipeline: architecture design, model construction, pre-trained weight loading, real-world image inference, and a live web application deployment.

---

## Research Foundation

This implementation is based on the paper:

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. (2015).
Rethinking the Inception Architecture for Computer Vision.
Google Inc., arXiv:1512.00567

---

## What Was Built

### 1. Custom Helper Function

A reusable conv2d_bn() function was implemented that combines a Conv2D layer, Batch Normalization, and ReLU Activation into a single modular unit. This function is used throughout the entire architecture and reflects the standard building block of InceptionV3.

### 2. Inception Block A — 35 x 35 Feature Maps

Three Inception A blocks were stacked at the 35 x 35 spatial resolution stage. Each block contains four parallel branches:

- A 1x1 convolution branch producing 64 filters
- A 1x1 followed by 5x5 convolution branch producing 64 filters
- A 1x1 followed by two 3x3 convolutions branch producing 96 filters
- An Average Pooling branch followed by a 1x1 convolution producing 32 filters

All four branches are concatenated along the channel axis, resulting in 256 feature maps.

### 3. Reduction Block A

This block reduces the spatial dimensions from 35 x 35 to 17 x 17 while increasing the number of channels to 736. It uses strided convolutions and max pooling in parallel branches to achieve dimensionality reduction without losing feature richness.

### 4. Inception Block B — 17 x 17 Feature Maps

Four Inception B blocks were stacked at the 17 x 17 stage. Each block uses asymmetric convolutions — factorizing standard 7x7 convolutions into 1x7 and 7x1 sequences — which reduces computation while maintaining representational power. The output of each block is 768 feature maps.

### 5. Reduction Block B

This block transitions from 17 x 17 to 8 x 8 spatial resolution, increasing the channel depth to 1280. It combines 3x3 strided convolutions, 7x7 asymmetric convolutions, and max pooling in parallel.

### 6. Inception Block C — 8 x 8 Feature Maps

Two Inception C blocks operate at the deepest spatial level. Each block expands branches further using parallel 1x3 and 3x1 convolutions, producing wide feature representations. The output depth reaches 2048 channels.

### 7. Classification Head

- Global Average Pooling reduces the 8 x 8 x 2048 tensor to a flat 2048-dimensional vector
- A Dense layer with 1000 units and Softmax activation produces the final class probability distribution over all 1000 ImageNet categories

---

## Model Statistics

- Total Parameters: 22,078,280
- Trainable Parameters: 22,045,512
- Non-trainable Parameters: 32,768
- Input Shape: 299 x 299 x 3
- Output Classes: 1000 (ImageNet)
- Model Size: approximately 84 MB

---

## Architecture Flow

Input 299x299x3
  --> Stem Conv Layers (35x35x192)
    --> Inception Block A x3 (35x35x256)
      --> Reduction Block A (17x17x736)
        --> Inception Block B x4 (17x17x768)
          --> Reduction Block B (8x8x1280)
            --> Inception Block C x2 (8x8x2048)
              --> Global Average Pooling (2048)
                --> Dense Softmax Output (1000 classes)

---

## Pre-Trained Model and Real-World Inference

After building the architecture from scratch, Google's official ImageNet pre-trained weights were loaded using Keras Applications. A real-world image was passed through the model pipeline:

- Image was resized to 299 x 299 pixels
- Pixel values were preprocessed using InceptionV3's normalization function
- The model returned Top-5 predictions with confidence scores
- Results were visualized using Matplotlib

The model correctly identified the dominant object class in the test image, confirming the implementation was functionally correct end-to-end.

---

## Web Application

A live web application was built and deployed using HTML, CSS, and JavaScript with TensorFlow.js. The application allows any user to upload an image from their device and receive real-time classification predictions directly in the browser — no server required. The app displays the top 5 predicted classes along with confidence scores and animated progress bars.

Live Demo: https://inceptionv3imageclassification.netlify.app

---

## Project Files

- inceptionV3.ipynb — Full scratch implementation of InceptionV3 architecture
- InceptionV3_Pre-Trained_Model.ipynb — Pre-trained model loading and inference
- InceptionV3_Pre-Trained_Model_with_prajwal_image.ipynb — Real image prediction demo
- index.html — Deployed web application

---

## Tech Stack

- Language: Python 3.12
- Deep Learning Framework: TensorFlow 2.x
- Model API: Keras
- Numerical Computing: NumPy
- Visualization: Matplotlib
- Web Deployment: HTML, CSS, JavaScript, TensorFlow.js
- Hosting: Netlify
- Development Environment: Jupyter Notebook

---

## Key Achievements

- Implemented a 22 million parameter production-grade neural network entirely from scratch without using any pre-built model class
- Faithfully reproduced all architectural components described in the original Google research paper
- Validated the implementation by loading official pre-trained weights and running successful inference on real-world images
- Deployed a fully functional, browser-based image classification web application accessible to anyone with an internet connection

---

Powered by Prajwal Ghotkar
