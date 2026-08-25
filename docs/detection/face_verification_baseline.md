# Baseline Face AI Stack: YuNet & SFace

## Overview
This document specifies the technical architecture, licensing, and baseline characteristics of the initial Face AI stack for the AI Proctoring system.

---

## Model Specifications & Licenses

### 1. Face Detection: YuNet
* **Exact Model File**: `face_detection_yunet_2023mar.onnx`
* **Size**: 232 KB
* **Architecture**: YuNet (Lightweight Convolutional Neural Network for real-time edge face detection)
* **Function**: Detects bounding boxes `[x, y, w, h]`, detection confidence score, and 5 facial landmarks (right eye, left eye, nose tip, right mouth corner, left mouth corner).
* **Official Repository**: [OpenCV Zoo - YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
* **License**: **Apache License 2.0**
* **Input**: BGR Image `(H, W, 3)` with dynamic input size.
* **Output**: `(N, 15)` matrix representing detected faces and landmarks.

### 2. Face Recognition & Verification: SFace
* **Exact Model File**: `face_recognition_sface_2021dec.onnx`
* **Size**: 38.7 MB
* **Architecture**: SFace (SphereFace-based deep convolutional network optimized for lightweight face recognition)
* **Function**: Crops and aligns detected faces to `112x112`, generates 128-dimensional L2-normalized feature embeddings, and computes cosine similarity or L2 Euclidean distance.
* **Official Repository**: [OpenCV Zoo - SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
* **License**: **Apache License 2.0**
* **Input**: `(112, 112, 3)` aligned facial crop.
* **Output**: `(1, 128)` feature embedding vector.

---

## Architectural Rationale

1. **Lightweight & High Efficiency**:
   * Total model footprint is under 40 MB.
   * YuNet runs in single-digit milliseconds on standard x86_64 CPUs, eliminating the need for dedicated GPU hardware for local testing.

2. **Native OpenCV Integration**:
   * Supported out-of-the-box in `opencv-contrib-python` via `cv2.FaceDetectorYN` and `cv2.FaceRecognizerSF`.
   * Requires no heavy deep learning frameworks (PyTorch, TensorFlow, or ONNX Runtime wrappers) during initial baseline stages.

3. **Permissive Licensing**:
   * Both YuNet and SFace models are distributed under the Apache-2.0 license, permitting research, modification, and commercial reuse in accordance with license terms.

4. **Reproducibility & Cross-Platform Parity**:
   * Identical behavior between local CPU development environments, Docker containers, and Google Colab GPU/CPU sessions.

---

## Validation Dataset Specifications

* **Dataset**: **Labeled Faces in the Wild (LFW)** Benchmark Subset
* **Origin**: University of Massachusetts Amherst
* **License**: Public Domain / Open Academic Research Use
* **Purpose**: Standard unconstrained face verification evaluation using positive (same person) and negative (different people) image pairs.
