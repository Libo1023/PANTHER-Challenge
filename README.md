# PANTHER Challenge: Pancreatic Tumor Segmentation

## Overview

This repository contains the implementation for the PANTHER (Pancreatic Tumor Heterogeneity) challenge, addressing automated pancreatic tumor segmentation from MRI images. The solution encompasses two distinct tasks targeting different MRI modalities and implements advanced deep learning techniques including semi-supervised learning, domain adaptation, and ensemble methods.

## Structure

This repository is organized into two main directories corresponding to the challenge tasks:

- **Task1**: Contains the complete pipeline for T1-weighted MRI tumor segmentation, including Docker containerization files, inference scripts, model architectures, and comprehensive training strategies
- **Task2**: Houses the T2-weighted MRI segmentation solution with its two-stage approach, domain adaptation mechanisms, and evaluation utilities

Each task directory includes Docker configuration files for containerized deployment, nnUNet model configurations, custom training scripts, and evaluation metrics implementation. 

## Task 1: T1-weighted MRI Tumor Segmentation

Task 1 addresses pancreatic tumor segmentation from T1-weighted abdominal MRI scans. The dataset comprises 91 labeled cases and 389 unlabeled cases, presenting a semi-supervised learning challenge.

### Approach and Methodology

The solution implements a multi-stage training pipeline with progressive refinement:

**Architecture**: The core architecture employs ResEncM (Residual Encoder Medium), a memory-efficient variant of the nnUNet framework. This architecture features residual connections in the encoder path, providing improved gradient flow while maintaining computational efficiency with approximately 9GB VRAM usage.

**Training**: The implementation follows a noisy student training paradigm with multiple rounds of pseudo-labeling:

1. **Initial Teacher Training**: A 5-fold cross-validation ensemble is trained on the 91 labeled cases using the ResEncM architecture. Each fold trains for 300 epochs with a learning rate of 0.005, employing both cross-entropy and Dice loss with weights of 1.0 and 1.5 respectively.

2. **Pseudo-Label Generation**: The trained teacher ensemble generates predictions for all 389 unlabeled cases. These predictions undergo filtering to exclude background-only samples, ensuring quality pseudo-labels for the student training phase.

3. **Student Model Training**: Student models are trained on the combined dataset of original labeled data and filtered pseudo-labels, totaling approximately 420 training samples after background-only exclusion. The student training extends to 800-1600 epochs with adjusted learning rates ranging from 0.003 to 0.005.

**Multi-class to Binary Conversion**: While the models are trained for 3-class segmentation (background, tumor, pancreas), the final evaluation extracts only the tumor class for binary segmentation metrics, aligning with the challenge requirements.

**Ensemble Inference**: The final inference combines predictions from all 5 folds using softmax averaging, providing robust segmentation results through model consensus.

## Task 2: T2-weighted MRI Tumor Segmentation

Task 2 focuses on pancreatic tumor segmentation from T2-weighted MRI scans, with a limited dataset of 50 labeled cases. This task is challenging due to the different image characteristics and smaller dataset size.

### Approach and Methodology

The solution implements a two-stage segmentation pipeline with domain adaptation:

**Stage 1 - Pancreas Localization**: The pipeline employs MRSegmentator, a pre-trained multi-organ segmentation model, to identify the pancreas region. The input images are resampled to 3x3x6mm spacing for optimal performance. The pancreas mask is extracted from the multi-organ predictions and used to define a region of interest with 30mm margins.

**Stage 2 - Tumor Segmentation**: The cropped pancreatic region undergoes tumor segmentation using a 3-fold nnUNet ensemble. This focused approach reduces computational requirements while improving segmentation accuracy by constraining the problem space.

**Domain Adaptation**: To leverage the larger Task 1 dataset, the implementation incorporates: 

- **Histogram Matching**: Task 1 images are transformed to match the intensity distribution of Task 2 images, computed from all 50 Task 2 samples
- **Combined Training**: A unified model is trained on the domain-adapted Task 1 data (both labeled and pseudo-labeled) combined with Task 2 training samples
- **Intensity Statistics Alignment**: The adaptation process preserves spatial information while aligning intensity characteristics between the two MRI modalities

**Final Reconstruction**: The cropped tumor predictions are mapped back to the original image space, maintaining spatial correspondence with the input T2-weighted MRI.

## Technical Implementation Details

### Model Architectures

The implementation utilizes several nnUNet variants with specific configurations:

- **ResEncM**: Features [32, 64, 128, 256, 320, 320] channels across 6 stages with residual encoder blocks
- **3D Full Resolution**: Processes patches of size 48x160x224 for Task 1 and 64x112x160 for Task 2
- **Instance Normalization**: Applied throughout the network for training stability
- **Deep Supervision**: Implemented during training with exponentially decreasing weights

### Training Optimizations

The training pipeline incorporates several optimization strategies:

- **Data Augmentation**: Controlled rotation augmentation limited to 20 degrees to preserve anatomical realism
- **Mixed Precision Training**: Utilized for memory efficiency and faster training
- **Adaptive Learning Rate**: Polynomial decay schedule with initial rates tailored to each training phase
- **Cross-Validation**: 5-fold cross-validation for Task 1 ensures robust model selection and reduces overfitting

### Evaluation Metrics

The implementation computes comprehensive segmentation metrics:

- **Volumetric Dice Coefficient**: Measures overall segmentation overlap
- **Surface Dice at 5mm**: Evaluates boundary accuracy with 5mm tolerance
- **95th Percentile Hausdorff Distance**: Assesses worst-case boundary errors
- **Mean Average Surface Distance**: Quantifies average boundary deviation
- **Tumor Burden RMSE**: Evaluates volumetric estimation accuracy

## Docker Containerization

Both tasks include complete Docker configurations for deployment:

- **Task 1**: Implements a custom inference pipeline with 5-fold ensemble prediction and 3-class to binary conversion
- **Task 2**: Integrates MRSegmentator and nnUNet in a sequential pipeline with automatic ROI extraction and reconstruction

The containers are built on PyTorch base images with CUDA support, ensuring GPU acceleration for inference. Each container includes all necessary dependencies, model weights, and preprocessing/postprocessing utilities for standalone deployment.
