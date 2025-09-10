
# PANTHER Challenge: Pancreatic Tumor Segmentation on MR

This repository contains the implementation for the PANTHER (PANcreatic Tumor sEgmentation on MR) challenge, which focuses on automated segmentation of pancreatic tumors from MRI scans using state-of-the-art deep learning approaches.

## Overview

The PANTHER challenge aims to develop robust algorithms for pancreatic tumor segmentation from different MRI sequences. This repository provides complete pipelines for both training and deployment of segmentation models across two distinct tasks with different imaging modalities.

## Tasks Description

### Task 1: Tumor Segmentation on T1-weighted MRI

Task 1 focuses on segmenting pancreatic tumors from T1-weighted abdominal MRI scans. The approach uses a 3-class segmentation strategy:
- Background (class 0)
- Tumor tissue (class 1)  
- Pancreas tissue (class 2)

**Key Features:**
- 5-fold cross-validation ensemble approach
- ResidualEncoderUNet architecture (ResEnc-M variant)
- 800 training epochs per fold
- Ensemble prediction combining all 5 folds using mean or majority voting
- Binary tumor extraction from multi-class predictions for final output

### Task 2: Tumor Segmentation on T2-weighted MRI

Task 2 addresses tumor segmentation on T2-weighted MR-Linac MRI scans using a two-stage approach that first localizes the pancreas region before performing detailed tumor segmentation.

**Key Features:**
- Two-stage segmentation pipeline
- MRSegmentator for initial pancreas localization at low resolution
- Cropping to pancreas ROI with 30mm margins
- Focused tumor segmentation within the cropped region
- 3-fold cross-validation
- Fine-tuning from Task 1 pre-trained models

## Repository Structure

```
.
├── Task1/
│   ├── Dockerfile                 # Container definition for Task 1
│   ├── inference.py              # 5-fold ensemble inference script
│   ├── custom_trainers/          # Custom nnU-Net trainer implementations
│   ├── nnUNet_results/           # Model configurations and plans
│   ├── requirements.txt          # Python dependencies
│   └── do_*.sh                   # Build, test, and save scripts
│
├── Task2/
│   ├── Dockerfile                # Container definition for Task 2
│   ├── inference.py             # Two-stage inference pipeline
│   ├── data_utils.py            # Image processing utilities
│   ├── nnUNet_results/          # Model configurations
│   ├── requirements.txt         # Python dependencies
│   └── do_*.sh                  # Build, test, and save scripts
│
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU with at least 12GB VRAM
- Docker for containerized deployment
- nnU-Net v2 framework

### Setting up the Environment

1. Clone the repository:
```bash
git clone <repository-url>
cd PANTHER-Challenge
```

2. Install dependencies for Task 1:
```bash
pip install nnunetv2 SimpleITK scipy
```

3. For Task 2, additionally install:
```bash
pip install mrsegmentator evalutils
```

4. Set nnU-Net environment variables:
```bash
export nnUNet_raw="./nnUNet_raw"
export nnUNet_preprocessed="./nnUNet_preprocessed"
export nnUNet_results="./nnUNet_results"
```

## Model Architecture

### ResidualEncoderUNet-M (ResEnc-M)

Both tasks utilize a ResidualEncoderUNet with the following architecture:
- 6 resolution stages with feature channels: [32, 64, 128, 256, 320, 320]
- Residual blocks for improved gradient flow
- Instance normalization and LeakyReLU activation
- Deep supervision for better convergence
- Patch size: 48 x 160 x 224 for 3D processing

The ResEnc-M variant provides an optimal balance between memory efficiency (9-11GB VRAM) and segmentation accuracy.

## Inference Pipeline

### Task 1: 5-Fold Ensemble Inference

The Task 1 inference script performs:
1. Loading of all 5 fold models from the ensemble
2. Independent inference for each fold on the input image
3. Ensemble aggregation using either mean probabilities or majority voting
4. Extraction of the tumor class (class 1) for binary output
5. Saving the final segmentation mask

### Task 2: Two-Stage Inference

The Task 2 inference pipeline consists of:
1. **Stage 1 - Pancreas Localization:**
   - Resample input to low resolution (3x3x6 mm spacing)
   - Run MRSegmentator to identify pancreas region
   - Extract pancreas mask (class 7 from MRSegmentator output)

2. **Stage 2 - Tumor Segmentation:**
   - Crop original image to pancreas bounding box with 30mm margins
   - Run nnU-Net on the cropped region
   - Restore predictions to original image space
   - Output binary tumor mask

## Docker Deployment

### Building Containers

For Task 1:
```bash
cd Task1
./do_build.sh panther-task1-5fold-ensemble
```

For Task 2:
```bash
cd Task2
./do_build.sh panther-task2-baseline
```

### Testing Containers Locally

Both tasks include test scripts to validate the container functionality:

```bash
# Task 1
cd Task1
./do_test_run.sh panther-task1-5fold-ensemble

# Task 2
cd Task2
./do_test_run.sh panther-task2-baseline
```

The test scripts will:
- Mount test input/output directories
- Run inference on sample data
- Verify output generation

### Preparing for Submission

To create submission-ready containers:

```bash
# Task 1
cd Task1
./do_save.sh panther-task1-5fold-ensemble

# Task 2
cd Task2
./do_save.sh panther-task2-baseline
```

This generates:
- Compressed Docker image (.tar.gz)
- Model weights tarball (if applicable)

## Input/Output Specifications

### Input Format
- File format: MHA (MetaImage)
- Image type: T1-weighted MRI (Task 1) or T2-weighted MRI (Task 2)
- Naming convention: Single MHA file in the input directory

### Output Format
- File format: MHA
- Content: Binary segmentation mask (0 = background, 1 = tumor)
- Location: `/output/images/pancreatic-tumor-segmentation/tumor_seg.mha`

## Evaluation Metrics

The models are evaluated using five complementary metrics:

1. **Volumetric Dice Score** - Overlap between predicted and ground truth
2. **Surface Dice Score** - Boundary accuracy with 5mm tolerance
3. **Hausdorff Distance (95%)** - Maximum boundary deviation
4. **Mean Average Surface Distance** - Average distance between surfaces
5. **Tumor Burden RMSE** - Volume estimation accuracy

## Key Implementation Details

### Task 1 Specifics
- Training uses 3-class segmentation but evaluation focuses on tumor class only
- 5-fold models are trained independently without information leakage
- Ensemble method can be configured (mean vs majority voting)
- Each fold uses identical architecture but different data splits

### Task 2 Specifics
- MRSegmentator weights must be available at `/opt/ml/model/weights`
- Pancreas cropping includes safety margins to avoid boundary effects
- Fine-tuning leverages Task 1 learned features for faster convergence
- Two-stage approach reduces computational requirements

## Notes

- Model checkpoints should be placed in the appropriate directories before building containers
- The inference scripts include comprehensive error handling and logging
- GPU acceleration is required for optimal performance
- Both tasks support the Grand Challenge platform specifications
