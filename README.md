# PANTHER Challenge: Pancreatic Tumor Segmentation
This repository contains my algorithms for the PANTHER Challenge, tackling both tasks. 

## Overview
The PANTHER Challenge focuses on automated segmentation of pancreatic tumors from abdominal MRI scans.  
My algorithms leverage nnU-Net with several advanced techniques including:  
- **Pseudo-labeling** to leverage unlabeled data
- **5-fold ensemble inference** for robust predictions
- **Noisy student training** for semi-supervised learning
- **ResEnc (Residual Encoder) U-Net architecture** variants
- **Pre-training and fine-tuning** for effective transfer between two tasks

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
└── Task2/                     # Pancreatic tumor segmentation in MR-Linac MRIs
    ├── Dockerfile             # Container definition for Task 2
    ├── inference.py           # Cropped inference with MRSegmentator
    ├── data_utils.py          # Image preprocessing utilities
    ├── nnUNet_results/        # Model configurations
    ├── training/              # Training scripts and utilities
    │   └── train.py           # 3-fold ResEncM fine-tuning
    └── model/                 # Checkpoint storage directory
```

## Methodology

### Task 1: Diagnostic MRI Segmentation
**Architecture:** ResEnc-M (Residual Encoder U-Net Medium variant) with 3-class output (background as label 0, tumor as label 1, and pancreas as label 2)

**Training Strategy - Two-Round Noisy Student Training:**
1. **Round 0 - Teacher Training:** 
   - Train 5-fold ResEnc-M models on 92 labeled samples
   - 300 epochs with initial LR of 0.005
   
2. **Round 1 - First Student Generation:**
   - Teachers generate pseudo-labels for 389 unlabeled samples
   - Filter out background-only predictions
   - Train 5-fold student models on combined labeled + pseudo-labeled data (800 epochs)

3. **Round 2 - Enhanced Student Generation:**
   - Round 1 students generate improved pseudo-labels
   - Train final 5-fold student models with 1600 epochs and LR of 0.003
   
**Inference:** 
- 5-fold ensemble with softmax averaging
- Multi-class predictions (3 classes) with binary tumor extraction
- Per-fold inference followed by probability averaging

### Task 2: MR-Linac MRI Segmentation
**Architecture:** ResEnc-M fine-tuned from Task 1 models

**Training Strategy:**
- 3-fold cross-validation on 50 labeled MR-Linac samples
- Fine-tune from Task 1 checkpoint (500 epochs, LR 0.001)
- Maintain 3-class segmentation framework

**Inference Pipeline:**
1. Resample input to low resolution (3.0, 3.0, 6.0 mm)
2. Apply MRSegmentator to detect pancreas region
3. Crop original image to pancreas ROI with 30mm margins
4. Run 3-fold ensemble nnU-Net inference on cropped region
5. Restore predictions to original image dimensions

## Key Features

### Semi-Supervised Learning
- **Pseudo-labeling:** Leverages 389 unlabeled samples in Task 1
- **Noisy Student Training:** Iterative refinement through teacher-student paradigm
- **Background Filtering:** Excludes pseudo-labels with no foreground to maintain quality

### Ensemble Methods
- **Task 1:** 5-fold cross-validation ensemble
- **Task 2:** 3-fold cross-validation ensemble
- **Softmax Averaging:** Probabilistic combination of fold predictions

### Domain Adaptation
- **Transfer Learning:** Task 2 models initialized from Task 1 weights
- **ROI-based Processing:** MRSegmentator-guided cropping for focused segmentation

## Requirements

### Dependencies
- PyTorch 2.3.1 with CUDA 11.8
- nnU-Net v2
- SimpleITK
- MRSegmentator (Task 2 only)
- surface-distance (for evaluation)

### Hardware
- GPU with ~9GB VRAM for ResEnc-M variant
- 24GB VRAM for ResEnc-L variant (optional)
- 40GB VRAM for ResEnc-XL variant (optional)

## Usage

### Docker Container Building

```bash
# Task 1
cd Task1
./do_build.sh panther-task1-5fold-ensemble
./do_test_run.sh  # Test with sample data
./do_save.sh  # Create submission package

# Task 2
cd Task2
./do_build.sh panther-task2-baseline
./do_test_run.sh
./do_save.sh
```

### Training

#### Task 1 Training
```python
# Round 0: Initial teacher training
python training/colab_training.py

# Round 1: Generate pseudo-labels and train students
python training/r1_pseudo_panther/generate_teacher_predictions.py
python training/r1_students_panther/train_students.py

# Round 2: Refined pseudo-labels and final students
python training/r2_pseudo_panther/generate_prediction.py
python training/r2_students_panther/train.py --fold 0  # Repeat for folds 0-4
```

#### Task 2 Training
```python
# Fine-tune from Task 1 (3 folds)
python training/train.py --fold 0 --task1_checkpoint path/to/checkpoint.pth
python training/train.py --fold 1 --task1_checkpoint path/to/checkpoint.pth
python training/train.py --fold 2 --task1_checkpoint path/to/checkpoint.pth
```

### Evaluation

Both tasks include evaluation scripts that compute:
- Volumetric Dice Score
- Surface Dice (5mm tolerance)
- 95% Hausdorff Distance
- Mean Average Surface Distance (MASD)
- Tumor Burden RMSE

```python
# Example evaluation
python training/evaluate_local_fixed.py \
    --pred_dir predictions/ \
    --gt_dir ground_truth/ \
    --verbose
```

## Model Configurations

### nnU-Net Plans
- **Network:** ResidualEncoderUNet
- **Normalization:** Z-Score
- **Patch Size:** 
  - Task 1: [48, 160, 224]
  - Task 2: [64, 112, 160]
- **Batch Size:** 2
- **Deep Supervision:** Enabled

### Training Hyperparameters
| Parameter | Task 1 Teacher | Task 1 Student | Task 2 |
|-----------|---------------|----------------|---------|
| Epochs | 300 | 800-1600 | 500 |
| Initial LR | 0.005 | 0.003-0.005 | 0.001 |
| Optimizer | SGD with momentum | SGD with momentum | SGD with momentum |
| Loss | CE + Dice (1:1.5) | CE + Dice | CE + Dice |
