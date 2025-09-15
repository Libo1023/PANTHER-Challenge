# PANTHER Challenge: Pancreatic Tumor Segmentation

This repository contains my algorithms for the PANTHER Challenge, addressing both Task 1 (T1-weighted MRI) and Task 2 (T2-weighted MRI) pancreatic tumor segmentation.

## Overview

The PANTHER Challenge focuses on automated segmentation of pancreatic tumors from abdominal MRI scans.  
My algorithms leverage nnU-Net with several advanced techniques including:  

- **Pseudo-labeling** to leverage unlabeled data
- **5-fold ensemble inference** for robust predictions
- **Noisy student training** for semi-supervised learning
- **ResEnc (Residual Encoder) U-Net architecture** variants
- **Pre-training and fine-tuning** for effective transfer between two tasks. 

## Repository Structure

```
.
├── Task1/                     # Pancreatic tumor segmentation in diagnostic MRIs
│   ├── Dockerfile             # Container definition for Task 1
│   ├── inference.py           # 5-fold ensemble inference pipeline
│   ├── custom_trainers/       # Custom nnU-Net trainer implementations
│   ├── nnUNet_results/        # Model configurations and plans
│   ├── training/              # Training scripts and utilities
│   │   ├── colab_training.py       # Round 0: 5-fold teacher model training
│   │   ├── r1_pseudo_panther/      # Round 1: 5-fold teacher model pseudo-labeling
│   │   ├── r1_students_panther/    # Round 1: 5-fold noisy student model training
│   │   ├── r2_pseudo_panther/      # Round 2: 5-fold student model pseudo-labeling
│   │   └── r2_students_panther/    # Round 2: Enhanced 5-fold noisy student model training
│   └── model/                 # Checkpoint storage directory
│
└── Task2/                     # Pancreatic tumor segmentation MR-Linac MRIs
│   ├── Dockerfile             # Container definition for Task 2
│   ├── inference.py           # Cropped inference with MRSegmentator
│   ├── data_utils.py          # Image preprocessing utilities
│   ├── nnUNet_results/        # Model configurations
│   └── training/              # Training scripts and utilities
│   │   ├── train.py           # 3-fold ResEncM fine-tuning
│   └── model/                 # Checkpoint storage directory
```

## Solution Approach

### Task 1: Pancreatic Tumor Segmentation in Diagnostic MRIs

Our Task 1 solution implements a multi-stage approach:

1. **5-Fold Ensemble Model**
   - ResEnc-M architecture (Residual Encoder U-Net Medium variant)
   - 800 epochs per fold training
   - Soft voting ensemble for final predictions
   - 3-class segmentation (background, tumor, pancreas) with binary output extraction

2. **Noisy Student Training Pipeline**
   - Teacher model generates pseudo-labels for unlabeled data
   - Student model trained on combined labeled and pseudo-labeled data
   - Background-only predictions filtered out
   - 1600 epochs for student training with learning rate 0.003

### Task 2: Pancreatic Tumor Segmentation MR-Linac MRIs

The Task 2 solution incorporates:

1. **Two-Stage Processing**
   - MRSegmentator for pancreas localization
   - nnU-Net for tumor segmentation on cropped ROI
   - Restoration to full image size

2. **Domain Adaptation**
   - Histogram matching to align Task 1 data to Task 2 intensity distribution
   - Combined training on adapted Task 1 and Task 2 data
   - Z-score normalization option available

## Key Features

### Inference Pipeline (Task 1)
- **Input Processing**: Automatic detection of MRI folder structure
- **Multi-fold Inference**: Parallel processing across 5 folds
- **Ensemble Method**: Softmax averaging for robust predictions
- **Output Format**: Binary tumor mask in MHA format

### Training Enhancements
- **Pseudo-Labeling**: Leverages ~350 unlabeled images
- **Foreground Filtering**: Excludes background-only pseudo-labels
- **Custom Learning Schedule**: Optimized for pancreatic tumor characteristics
- **Progress Monitoring**: Detailed logging and evaluation metrics

### Evaluation Metrics
The solution is evaluated using five key metrics:
1. Volumetric Dice Coefficient
2. Surface Dice at 5mm tolerance
3. 95th percentile Hausdorff Distance
4. Mean Average Surface Distance (MASD)
5. Tumor Burden RMSE

## Docker Deployment

### Building the Container

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

### Running Inference

Task 1 test run:
```bash
cd Task1
./do_test_run.sh panther-task1-5fold-ensemble
```

Task 2 test run:
```bash
cd Task2
./do_test_run.sh panther-task2-baseline
```

### Saving for Submission
```bash
./do_save.sh [container-name]
```

This creates a compressed container image ready for Grand Challenge submission.

## Training Scripts

### Task 1 Training Pipeline

1. **Teacher Model Training** (Google Colab):
```python
python Task1/training/google-colab_5-fold_pretraining.py
```
- Trains 5-fold ResEnc-M model
- 300 epochs per fold
- Generates initial predictions

2. **Pseudo-Label Generation**:
```python
python Task1/training/r1_pseudo_panther/generate_teacher_predictions.py
```
- Creates 3-class predictions for unlabeled data
- 5-fold ensemble inference

3. **Student Model Training**:
```python
python Task1/training/r1_students_panther/train_students.py
```
- Combines labeled and pseudo-labeled data
- Extended training (800-1600 epochs)

### Task 2 Training with Domain Adaptation

```python
python Task2/training/train.py
```
- Applies histogram matching for domain adaptation
- Combines Task 1 and Task 2 data
- Trains from scratch with adapted data

## Configuration

Key configuration parameters can be adjusted in the training scripts:

```python
CONFIG = {
    "dataset_id": 91,
    "resenc_variant": "M",  # Options: "M", "L", "XL"
    "num_epochs": 800,
    "initial_lr": 0.005,
    "ensemble_method": "mean",  # or "majority_vote"
    "surface_tolerance_mm": 5,
    "hausdorff_percentile": 95
}
```

## Data Organization

Expected data structure:
```
data/
├── PANTHER_Task1/
│   ├── ImagesTr/           # Labeled T1 images
│   ├── LabelsTr/           # Ground truth labels
│   └── ImagesTr_unlabeled/ # Unlabeled T1 images
└── PANTHER_Task2/
    ├── ImagesTr/           # T2 images
    └── LabelsTr/           # Ground truth labels
```

## Model Architectures

### ResEnc Variants
- **ResEnc-M**: ~9GB VRAM, balanced performance
- **ResEnc-L**: ~24GB VRAM, enhanced capacity
- **ResEnc-XL**: ~40GB VRAM, maximum performance

### Network Configuration (3D)
- Patch size: 48×160×224 (Task 1)
- Batch size: 2
- Instance normalization
- LeakyReLU activation
- Deep supervision with weighted losses

## Performance Considerations

### Hardware Requirements
- **Minimum**: GPU with 12GB VRAM (ResEnc-M)
- **Recommended**: GPU with 24GB+ VRAM for larger variants
- **Training Time**: ~12 hours per fold on A100 (ResEnc-M)

### Memory Optimization
- Checkpoint loading on-demand
- Temporary file cleanup after inference
- Efficient ensemble voting implementation

## Evaluation

Run local evaluation:
```python
python evaluate_local_fixed.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truth \
    --verbose
```

The evaluation script computes all PANTHER metrics including surface-based measurements using the DeepMind surface-distance library.

## Dependencies

Core requirements:
- PyTorch 2.3.1 with CUDA 11.8
- nnU-Net v2
- SimpleITK
- NumPy
- SciPy
- surface-distance
- mrsegmentator (Task 2 only)

See `requirements.txt` in each task folder for complete dependencies.
