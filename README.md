
# PANTHER Challenge - Pancreatic Tumor Segmentation

## Overview

This repository contains our solution for the PANTHER (Pancreatic Tumor Segmentation) Challenge, featuring automated segmentation of pancreatic tumors from MRI scans using deep learning approaches.

### Key Features
- **Task 1**: 5-fold ensemble segmentation for standard abdominal MRI (T1-weighted)
- **Task 2**: 3-fold segmentation for MR-Linac T2-weighted MRI with pancreas localization
- ResidualEncoder UNet architecture with multiple variants (M, L, XL)
- Docker containers for easy deployment
- Comprehensive evaluation metrics

## Repository Structure

```
.
├── Task1/                      # Standard MRI segmentation
│   ├── inference.py           # 5-fold ensemble inference
│   ├── custom_trainers/       # Custom nnUNet trainer
│   ├── Dockerfile             # Container for deployment
│   └── train_task1_demo.py    # Training pipeline (demo/reference)
│
├── Task2/                      # MR-Linac MRI segmentation  
│   ├── inference.py           # Two-stage inference
│   ├── data_utils.py          # Cropping and restoration utilities
│   ├── Dockerfile             # Container for deployment
│   └── train_task2_demo.py    # Fine-tuning pipeline (demo/reference)
│
└── evaluation/                 # Evaluation scripts
```

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (9GB+ VRAM for ResEnc-M)
- Docker (for deployment)
- nnU-Net v2

### Installation

```bash
# Clone repository
git clone https://github.com/Libo1023/PANTHER-Challenge/tree/main
cd panther-challenge

# Install dependencies
pip install nnunetv2
pip install SimpleITK scipy surface-distance
pip install mrsegmentator  # For Task 2 only
```

## Training

**Note**: The training scripts (`train_task1_demo.py` and `train_task2_demo.py`) are provided as reference implementations demonstrating our training approach. They require adaptation to your specific environment and data paths.

### Task 1: Standard MRI (5-fold ensemble)

Required data structure:
```
data/PANTHER_Task1/
  ├── ImagesTr/  (92 .mha files)
  └── LabelsTr/  (92 .mha files)
```

**Configuration options** in `train_task1_demo.py`:
- `resenc_variant`: "M" (9GB), "L" (24GB), or "XL" (40GB)
- `num_epochs`: 800 (default)
- `num_folds`: 5 (cross-validation)

The training pipeline includes:
- Dataset creation with nnU-Net format
- Custom trainer with 3-class segmentation
- 5-fold cross-validation setup
- Automatic evaluation after each fold

### Task 2: MR-Linac MRI (3-fold with fine-tuning)

Required data structure:
```
data/PANTHER_Task2/
  ├── ImagesTr/  (50 .mha files)
  └── LabelsTr/  (50 .mha files)
```

The fine-tuning approach:
- Loads pretrained weights from Task 1
- Adapts to Task 2 data with reduced epochs (500)
- Uses 3-fold cross-validation
- Requires path to Task 1 checkpoint

## Docker Deployment

### Build containers

```bash
# Task 1
cd Task1
./do_build.sh panther-task1-ensemble

# Task 2  
cd Task2
./do_build.sh panther-task2-baseline
```

### Test locally

```bash
# Place test image in test/input/
./do_test_run.sh

# Output will be in test/output/
```

### Save for upload

```bash
./do_save.sh
# Creates .tar.gz for Grand Challenge upload
```

## Model Architecture

### ResidualEncoder UNet Variants

| Variant | VRAM Usage | Training Time/Fold | Performance |
|---------|------------|-------------------|-------------|
| ResEnc-M | ~9 GB | ~12 hours | Baseline |
| ResEnc-L | ~24 GB | ~35 hours | +2-3% Dice |
| ResEnc-XL | ~40 GB | ~66 hours | +3-5% Dice |

### Training Details

**Task 1**:
- 3-class segmentation (background, tumor, pancreas)
- Binary evaluation (tumor only)
- 800 epochs, learning rate: 0.005
- 5-fold cross-validation with ensemble

**Task 2**:
- Fine-tuned from Task 1
- Pancreas localization → tumor segmentation
- 500 epochs, learning rate: 0.001
- 3-fold cross-validation

## Evaluation Metrics

The models are evaluated using:
1. **Volumetric Dice** - Overlap coefficient
2. **Surface Dice** (5mm tolerance) - Boundary accuracy
3. **Hausdorff Distance 95%** - Maximum surface distance
4. **Mean Average Surface Distance (MASD)** - Average boundary error
5. **Tumor Burden RMSE** - Volume estimation accuracy

## Results

### Task 1 Performance (ResEnc-M, 5-fold average)
- Volumetric Dice: 0.XX ± 0.XX
- Surface Dice: 0.XX ± 0.XX
- HD95: XX.X ± X.X mm
- MASD: X.X ± X.X mm

### Task 2 Performance (3-fold average)
- Volumetric Dice: 0.XX ± 0.XX
- Surface Dice: 0.XX ± 0.XX
- HD95: XX.X ± X.X mm
- MASD: X.X ± X.X mm

## Key Implementation Details

### Task 1: 5-Fold Ensemble
- Trains 5 independent models on different data splits
- Ensemble via soft voting (averaging probabilities)
- Extracts tumor class from 3-class predictions

### Task 2: Two-Stage Pipeline
1. **Pancreas Localization**: MRSegmentator at low resolution
2. **ROI Cropping**: Extract pancreas region with 30mm margin
3. **Tumor Segmentation**: nnUNet on cropped region
4. **Restoration**: Map predictions back to original space
