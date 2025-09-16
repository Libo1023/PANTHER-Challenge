# PANTHER Challenge: Pancreatic Tumor Segmentation
This repository contains algorithmic solutions for the PANTHER Challenge, tackling both tasks. 

## Overview
The PANTHER Challenge focuses on automated segmentation of pancreatic tumors from abdominal MRI scans.  
The algorithms leverage nnU-Net with several advanced techniques including:  
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
   - Teachers generate pseudo-labels for 367 unlabeled samples
   - Filter out background-only predictions
   - Train 5-fold student models on combined labeled + pseudo-labeled data (800 epochs)

3. **Round 2 - Enhanced Student Generation:**
   - Round 1 students generate improved pseudo-labels
   - Filter out background-only predictions (fewer than Round 1)
   - Train final 5-fold student models with 1600 epochs and LR of 0.003
   
**Inference:** 
- 5-fold ensemble with softmax probability averaging
- Multi-class predictions (3 classes) with binary tumor extraction

### Task 2: MR-Linac MRI Segmentation
**Architecture:** ResEnc-M fine-tuned from Task 1 models

**Training**
- 3-fold cross-validation on 50 labeled MR-Linac samples
- Fine-tune from one of 5-fold Task 1 checkpoints (500 epochs, LR 0.001)
- Maintain 3-class segmentation formulation for harder per-fold training

**Inference**

1. **Input Processing**: 
   - Load original high-resolution T2-weighted MRI
   - Create low-resolution copy with 3.0×3.0×6.0 mm spacing for computational efficiency

2. **Organ Detection**: 
   - Apply MRSegmentator with **5-fold ensemble** (folds 0-4) on the low-resolution image
   - Extract pancreas segmentation mask (organ class #7 in MRSegmentator output)
   - Binary conversion: pancreas=1, everything else=0

3. **ROI Definition**: 
   - Compute 3D bounding box around the detected pancreas region
   - Add 30mm safety margins in all directions to ensure complete tumor coverage
   - Transform coordinates back to original image space

4. **Focused Processing**: 
   - Crop the **original high-resolution MRI** using the computed ROI (preserving original resolution)
   - This reduces the input volume by ~70-80%, enabling efficient processing

5. **Tumor Detection**: 
   - Run nnU-Net **3-fold ensemble** (folds 0, 1, 2) on the cropped high-resolution region
   - Model trained specifically for pancreatic tumor segmentation in T2-weighted MR images

6. **Full Resolution Reconstruction**: 
   - Map the predicted tumor mask from cropped space back to original image dimensions
   - Place predictions at the correct anatomical location using saved crop coordinates
   - Output final binary mask (0=background, 1=tumor) at original resolution

## Key Features

### Semi-Supervised Learning
- **Pseudo-labeling:** Leverages 367 unlabeled samples in Task 1
- **Noisy Student Training:** Iterative refinement through teacher-student paradigm
- **Background Filtering:** Excludes pseudo-labels with no foreground to maintain quality

### Ensemble Methods
- **Task 1:** 5-fold cross-validation ensemble
- **Task 2:** 3-fold cross-validation ensemble
- **Softmax Averaging:** Probabilistic combination of fold predictions

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

Note: I did not observe explicit performance boost by replacing ResEncM with the L or XL variant for neither task. My interpretation is that the dataset size is relatively small so larger models tend to be underfit. 

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
# Round 0: Initial teacher training (implemented on Google CoLab)
# Refer to ./Task1/training/colab_training.py

# Round 1: Generate pseudo-labels and train students
python ./Task1/training/r1_pseudo_panther/generate_teacher_predictions.py
python ./Task1/training/r1_students_panther/train_students.py

# Round 2: Refined pseudo-labels and final students
python ./Task1/training/r2_pseudo_panther/generate_prediction.py
python ./Task1/training/r2_students_panther/train.py --fold 0  # Repeat for folds 0-4
```

#### Task 2 Training
```python
# Fine-tune from Task 1 (3 folds)
python ./Task2/training/train.py --fold 0 --task1_checkpoint path/to/checkpoint.pth
python ./Task2/training/train.py --fold 1 --task1_checkpoint path/to/checkpoint.pth
python ./Task2/training/train.py --fold 2 --task1_checkpoint path/to/checkpoint.pth
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
python ./Task2/training/evaluate_local_fixed.py \
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
