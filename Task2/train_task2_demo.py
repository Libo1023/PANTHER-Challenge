import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import shutil
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import subprocess
import sys
import argparse
import torch

# Import the FIXED evaluation function
sys.path.append('./')
from evaluate_local_fixed import evaluate_segmentation_performance

# ============================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================
parser = argparse.ArgumentParser(description='Fine-tuning Pipeline from Task 1 to Task 2 - Single Fold')
parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2],
                    help='Which fold to train (0-2)')
parser.add_argument('--task1_checkpoint', type=str, 
                    default='./nnUNet_results/Dataset091_PantherStudent/nnUNetTrainer_Student__nnUNetResEncUNetMPlans__3d_fullres/fold_0/checkpoint_best.pth',
                    help='Path to Task 1 checkpoint to load for fine-tuning')
args = parser.parse_args()

SELECTED_FOLD = args.fold
TASK1_CHECKPOINT = args.task1_checkpoint

# ============================================
# REQUIRED FOLDER STRUCTURE (YOU NEED TO PREPARE)
# ============================================
"""
REQUIRED FOLDER STRUCTURE BEFORE RUNNING THIS SCRIPT:

./
├── data/
│   └── PANTHER_Task2/
│       ├── ImagesTr/         # Task 2 training images (50 .mha files)
│       └── LabelsTr/         # Task 2 training labels (50 .mha files)
├── nnUNet_results/
│   └── Dataset091_PantherStudent/
│       └── nnUNetTrainer_Student__nnUNetResEncUNetMPlans__3d_fullres/
│           └── fold_0/
│               └── checkpoint_best.pth  # Pre-trained model from Task 1
└── evaluate_local_fixed.py   # Evaluation script

"""

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    # Dataset configuration for Task 2
    "dataset_id": 92,  # Using 92 for Task 2
    "dataset_name": "Dataset092_PANTHERTask2",
    
    # RESIDUAL ENCODER VARIANT SELECTION (same as Task 1)
    "resenc_variant": "M",  # Using ResEnc-M as in Task 1
    
    # Fine-tuning hyperparameters
    "num_epochs": 500,  # Reduced from 2000 in Task 1
    "initial_lr": 0.001,  # Same as Task 1
    
    # Training configuration - SINGLE FOLD
    "fold_to_train": SELECTED_FOLD,
    "trainer_name": "nnUNetTrainerFineTuneTask2",  # Custom trainer for fine-tuning
    
    # Base directory
    "base_dir": ".",
    
    # Data paths for Task 2
    "task2_images": "./data/PANTHER_Task2/ImagesTr/",
    "task2_labels": "./data/PANTHER_Task2/LabelsTr/",
    
    # Task 1 checkpoint path
    "task1_checkpoint": TASK1_CHECKPOINT,
    
    # Number of folds for Task 2
    "num_folds": 3
}

# ResEnc configurations (same as Task 1)
RESENC_CONFIGS = {
    "M": {
        "planner_name": "nnUNetPlannerResEncM",
        "plans_name": "nnUNetResEncUNetMPlans",
        "expected_vram": "~9 GB"
    }, 
    "L": {
        "planner_name": "nnUNetPlannerResEncL", 
        "plans_name": "nnUNetResEncUNetLPlans",
        "expected_vram": "~24 GB"
    }
}

CONFIG.update({
    "planner_name": RESENC_CONFIGS[CONFIG["resenc_variant"]]["planner_name"],
    "plans_name": RESENC_CONFIGS[CONFIG["resenc_variant"]]["plans_name"],
    "expected_vram": RESENC_CONFIGS[CONFIG["resenc_variant"]]["expected_vram"]
})

print(f"Fine-tuning Pipeline: Task 1 to Task 2 - FOLD {SELECTED_FOLD}")
print(f"ResEnc variant: {CONFIG['resenc_variant']}")
print(f"Expected VRAM usage: {CONFIG['expected_vram']}")
print(f"Fine-tuning epochs: {CONFIG['num_epochs']}")
print(f"Initial learning rate: {CONFIG['initial_lr']}")
print(f"Task 1 checkpoint: {CONFIG['task1_checkpoint']}")

# ============================================
# Set Environment Variables
# ============================================
base_dir = CONFIG["base_dir"]

os.environ['nnUNet_raw'] = os.path.join(base_dir, "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = os.path.join(base_dir, "nnUNet_preprocessed")
os.environ['nnUNet_results'] = os.path.join(base_dir, "nnUNet_results")

# Create necessary directories
os.makedirs(os.path.join(base_dir, "nnUNet_raw"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "nnUNet_preprocessed"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "nnUNet_results"), exist_ok=True)
os.makedirs(os.path.join(base_dir, f"evaluation_results_task2_fold{SELECTED_FOLD}"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "custom_trainers"), exist_ok=True)

print("\nEnvironment variables set:")
print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
print(f"nnUNet_results: {os.environ['nnUNet_results']}")

# ============================================
# Helper Functions
# ============================================
def convert_3class_to_binary(prediction_path, output_path):
    """Convert 3-class prediction (0,1,2) to binary (0,1) where 0,2->0 and 1->1"""
    img = sitk.ReadImage(prediction_path)
    arr = sitk.GetArrayFromImage(img)
    
    # Convert: 0 (background) -> 0, 1 (tumor) -> 1, 2 (pancreas) -> 0
    binary_arr = np.zeros_like(arr)
    binary_arr[arr == 1] = 1
    
    # Create new image with same metadata
    binary_img = sitk.GetImageFromArray(binary_arr)
    binary_img.CopyInformation(img)
    
    sitk.WriteImage(binary_img, output_path)
    return output_path

def generate_predictions_for_fold(fold, dataset_name, plans_name, trainer_name, output_dir, val_cases):
    """Generate predictions for validation cases of a specific fold"""
    
    # Create temporary directory for validation images
    temp_val_dir = os.path.join(CONFIG['base_dir'], f"temp_val_fold_{fold}")
    if os.path.exists(temp_val_dir):
        shutil.rmtree(temp_val_dir)
    os.makedirs(temp_val_dir, exist_ok=True)
    
    # Copy validation images
    raw_data_dir = os.path.join(CONFIG['base_dir'], "nnUNet_raw", dataset_name, "imagesTr")
    copied_count = 0
    for case in val_cases:
        src_file = os.path.join(raw_data_dir, f"{case}_0000.mha")
        if os.path.exists(src_file):
            shutil.copy2(src_file, temp_val_dir)
            copied_count += 1
    
    print(f"Copied {copied_count} validation images to temporary directory")
    
    # Run prediction
    pred_cmd = [
        "nnUNetv2_predict",
        "-i", temp_val_dir,
        "-o", output_dir,
        "-d", str(CONFIG['dataset_id']),
        "-p", plans_name,
        "-c", "3d_fullres",
        "-f", str(fold),
        "-tr", trainer_name
    ]
    
    print(f"Running prediction command: {' '.join(pred_cmd)}")
    result = subprocess.run(pred_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Prediction failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
    
    # Clean up
    shutil.rmtree(temp_val_dir)
    
    return result.returncode == 0

def evaluate_fold_predictions_with_conversion(pred_dir, gt_dir, val_cases, fold):
    """Evaluate predictions for a specific fold after converting 3-class to binary"""
    
    # Create directories for binary predictions and ground truth
    binary_pred_dir = os.path.join(CONFIG['base_dir'], f"temp_binary_pred_fold_{fold}")
    temp_gt_dir = os.path.join(CONFIG['base_dir'], f"temp_gt_fold_{fold}")
    
    # Clean up if exists
    if os.path.exists(binary_pred_dir):
        shutil.rmtree(binary_pred_dir)
    if os.path.exists(temp_gt_dir):
        shutil.rmtree(temp_gt_dir)
    
    os.makedirs(binary_pred_dir, exist_ok=True)
    os.makedirs(temp_gt_dir, exist_ok=True)
    
    # Convert 3-class predictions to binary
    print(f"Converting 3-class predictions to binary...")
    converted_count = 0
    for pred_file in Path(pred_dir).glob("*.mha"):
        case_id = pred_file.stem
        if case_id in val_cases:
            binary_output = os.path.join(binary_pred_dir, pred_file.name)
            convert_3class_to_binary(str(pred_file), binary_output)
            converted_count += 1
    
    print(f"Converted {converted_count} predictions to binary format")
    
    # Copy ground truth labels for validation cases
    raw_labels_dir = os.path.join(CONFIG['base_dir'], "nnUNet_raw", CONFIG['dataset_name'], "labelsTr")
    copied_count = 0
    for case in val_cases:
        src_file = os.path.join(raw_labels_dir, f"{case}.mha")
        if os.path.exists(src_file):
            # Also convert GT to binary (in case it has 3 classes)
            dest_file = os.path.join(temp_gt_dir, f"{case}.mha")
            img = sitk.ReadImage(src_file)
            arr = sitk.GetArrayFromImage(img)
            binary_arr = np.zeros_like(arr)
            binary_arr[arr == 1] = 1
            binary_img = sitk.GetImageFromArray(binary_arr)
            binary_img.CopyInformation(img)
            sitk.WriteImage(binary_img, dest_file)
            copied_count += 1
    
    print(f"Copied and converted {copied_count} ground truth labels for evaluation")
    
    # Run evaluation
    try:
        results = evaluate_segmentation_performance(
            pred_dir=binary_pred_dir,
            gt_dir=temp_gt_dir,
            subject_list=val_cases,
            verbose=True
        )
        
        # Clean up
        shutil.rmtree(binary_pred_dir)
        shutil.rmtree(temp_gt_dir)
        
        return results
    except Exception as e:
        print(f"ERROR evaluating fold {fold}: {e}")
        # Clean up
        if os.path.exists(binary_pred_dir):
            shutil.rmtree(binary_pred_dir)
        if os.path.exists(temp_gt_dir):
            shutil.rmtree(temp_gt_dir)
        return None

# ============================================
# Define Detailed 3-Fold Splits for Task 2
# ============================================
TASK2_SPLITS = [
    {  # Fold 0: 16 validation, 34 training
        "train": [
            "10317", "10318", "10321", "10322", "10323", "10324", "10325", "10327", 
            "10328", "10329", "10330", "10331", "10333", "10334", "10336", "10338", 
            "10339", "10340", "10341", "10342", "10343", "10344", "10346", "10348", 
            "10352", "10353", "10354", "10355", "10358", "10359", "10361", "10362", 
            "10364", "10365"
        ],
        "val": [
            "10303", "10304", "10306", "10307", "10309", "10310", "10312", "10314", 
            "10315", "10368", "10372", "10373", "10379", "10380", "10381", "10383"
        ]
    },
    {  # Fold 1: 17 validation, 33 training
        "train": [
            "10303", "10304", "10306", "10307", "10309", "10310", "10312", "10314", 
            "10315", "10368", "10372", "10373", "10379", "10380", "10381", "10383",
            "10344", "10346", "10348", "10352", "10353", "10354", "10355", "10358", 
            "10359", "10361", "10362", "10364", "10365", "10343", "10342", "10341", 
            "10340"
        ],
        "val": [
            "10317", "10318", "10321", "10322", "10323", "10324", "10325", "10327", 
            "10328", "10329", "10330", "10331", "10333", "10334", "10336", "10338", 
            "10339"
        ]
    },
    {  # Fold 2: 17 validation, 33 training
        "train": [
            "10303", "10304", "10306", "10307", "10309", "10310", "10312", "10314", 
            "10315", "10368", "10372", "10373", "10379", "10380", "10381", "10383",
            "10317", "10318", "10321", "10322", "10323", "10324", "10325", "10327", 
            "10328", "10329", "10330", "10331", "10333", "10334", "10336", "10338", 
            "10339"
        ],
        "val": [
            "10340", "10341", "10342", "10343", "10344", "10346", "10348", "10352", 
            "10353", "10354", "10355", "10358", "10359", "10361", "10362", "10364", 
            "10365"
        ]
    }
]

# Print detailed splits information
print("\n" + "="*60)
print("DETAILED 3-FOLD SPLITS FOR TASK 2")
print("="*60)
for fold_idx, split in enumerate(TASK2_SPLITS):
    print(f"\nFold {fold_idx}:")
    print(f"  Training samples ({len(split['train'])}): {len(split['train'])} cases")
    print(f"  Validation samples ({len(split['val'])}): {len(split['val'])} cases")
    
    # Verify no overlap
    overlap = set(split['train']) & set(split['val'])
    assert len(overlap) == 0, f"Fold {fold_idx} has overlap: {overlap}"
    
    # Verify total
    total = len(split['train']) + len(split['val'])
    assert total == 50, f"Fold {fold_idx} doesn't sum to 50! Got {total}"

print(f"\nSelected fold {SELECTED_FOLD} for training:")
print(f"  Training: {len(TASK2_SPLITS[SELECTED_FOLD]['train'])} cases")
print(f"  Validation: {len(TASK2_SPLITS[SELECTED_FOLD]['val'])} cases")

# ============================================
# Check if Dataset Already Exists
# ============================================
dataset_dir = os.path.join(base_dir, "nnUNet_raw", CONFIG['dataset_name'])
dataset_json_path = os.path.join(dataset_dir, "dataset.json")
preprocessed_dataset_dir = os.path.join(base_dir, "nnUNet_preprocessed", CONFIG['dataset_name'])
splits_path = os.path.join(preprocessed_dataset_dir, "splits_final.json")

# Flag to check if we need to create dataset
need_dataset_creation = not os.path.exists(dataset_json_path)
need_preprocessing = not os.path.exists(splits_path)

if need_dataset_creation:
    print("\n" + "="*60)
    print("CREATING TASK 2 DATASET STRUCTURE (First time setup)")
    print("="*60)
    
    # ============================================
    # Create Simple Custom Fine-tuning Trainer
    # ============================================
    finetune_trainer_code = f'''
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch

class {CONFIG["trainer_name"]}(nnUNetTrainer):
    """Minimal trainer for fine-tuning - only changes epochs and learning rate"""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # Only override epochs and learning rate for fine-tuning
        self.num_epochs = {CONFIG["num_epochs"]}
        self.initial_lr = {CONFIG["initial_lr"]}
        
    def on_train_epoch_start(self):
        """Log training progress"""
        super().on_train_epoch_start()
        if self.current_epoch % 50 == 0:
            self.print_to_log_file(f"Fine-tuning Task 2 - Epoch {{self.current_epoch}}/{{self.num_epochs}}")
'''

    # Save custom trainer
    trainer_path = os.path.join(base_dir, "custom_trainers", f"{CONFIG['trainer_name']}.py")
    with open(trainer_path, 'w') as f:
        f.write(finetune_trainer_code)
    print(f"\nSimple fine-tuning trainer created at: {trainer_path}")
    
    # Copy trainer to site-packages
    import site
    site_packages = site.getsitepackages()
    for sp in site_packages:
        nnunet_trainer_dir = os.path.join(sp, "nnunetv2/training/nnUNetTrainer")
        if os.path.exists(nnunet_trainer_dir):
            shutil.copy2(trainer_path, nnunet_trainer_dir)
            print(f"Copied trainer to: {nnunet_trainer_dir}")
            break
    
    # ============================================
    # Create Dataset Structure for Task 2
    # ============================================
    os.makedirs(os.path.join(dataset_dir, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labelsTr"), exist_ok=True)
    
    dest_images = os.path.join(dataset_dir, "imagesTr")
    dest_labels = os.path.join(dataset_dir, "labelsTr")
    
    # Get all unique cases from splits
    all_cases = set()
    for split in TASK2_SPLITS:
        all_cases.update(split['train'])
        all_cases.update(split['val'])
    all_cases = sorted(list(all_cases))
    
    # Copy Task 2 images and labels
    print("\nCopying Task 2 data...")
    copied_count = 0
    missing_files = []
    
    for case_id in all_cases:
        # Copy image
        src_img = os.path.join(CONFIG["task2_images"], f"{case_id}_0000.mha")
        dest_img = os.path.join(dest_images, f"{case_id}_0000.mha")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dest_img)
        else:
            missing_files.append(src_img)
            
        # Copy label
        src_label = os.path.join(CONFIG["task2_labels"], f"{case_id}.mha")
        dest_label = os.path.join(dest_labels, f"{case_id}.mha")
        if os.path.exists(src_label):
            shutil.copy2(src_label, dest_label)
            copied_count += 1
        else:
            missing_files.append(src_label)
    
    print(f"Copied {copied_count} Task 2 samples")
    if missing_files:
        print(f"WARNING: {len(missing_files)} files not found:")
        for f in missing_files[:5]:  # Show first 5
            print(f"  - {f}")
    
    # ============================================
    # Create dataset.json
    # ============================================
    dataset_json = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": {
            "background": 0,
            "tumor": 1,
            "pancreas": 2
        },
        "numTraining": len(all_cases),
        "file_ending": ".mha",
        "description": "PANTHER Task 2: MR-Linac MRIs (T2-weighted)"
    }
    
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"Created dataset.json with {len(all_cases)} training cases")
    
    # ============================================
    # Run Planning and Preprocessing
    # ============================================
    print("\n" + "="*60)
    print(f"Running nnU-Net planning with ResEnc{CONFIG['resenc_variant']}...")
    print("="*60)
    
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(CONFIG["dataset_id"]), "-pl", CONFIG["planner_name"], "--verify_dataset_integrity"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Planning failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    
    # ============================================
    # Create Custom 3-Fold Splits
    # ============================================
    print("\nCreating custom 3-fold splits for Task 2...")
    
    # Save splits
    os.makedirs(preprocessed_dataset_dir, exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(TASK2_SPLITS, f, indent=2)
    
    print(f"Saved 3-fold splits to: {splits_path}")
    
    # Save dataset info
    dataset_info = {
        "task": "Task 2",
        "description": "MR-Linac T2-weighted MRIs",
        "num_cases": len(all_cases),
        "num_folds": 3,
        "pretrained_from": "Task 1 Fold 0"
    }
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=4)

else:
    print("\n" + "="*60)
    print("DATASET ALREADY EXISTS - SKIPPING CREATION")
    print("="*60)
    
    # Load existing splits
    with open(splits_path, "r") as f:
        TASK2_SPLITS = json.load(f)

# ============================================
# Fine-tuning Phase for Selected Fold
# ============================================
print("\n" + "="*80)
print(f"FINE-TUNING TASK 2 - FOLD {SELECTED_FOLD}")
print("="*80)
print(f"Configuration:")
print(f"  Architecture: ResEnc{CONFIG['resenc_variant']}")
print(f"  Epochs: {CONFIG['num_epochs']}")
print(f"  Learning Rate: {CONFIG['initial_lr']}")
print(f"  Training samples: {len(TASK2_SPLITS[SELECTED_FOLD]['train'])}")
print(f"  Validation samples: {len(TASK2_SPLITS[SELECTED_FOLD]['val'])}")
print(f"  Pretrained checkpoint: {CONFIG['task1_checkpoint']}")
print("="*80)

# Check if checkpoint exists
if not os.path.exists(CONFIG['task1_checkpoint']):
    print(f"WARNING: Task 1 checkpoint not found at: {CONFIG['task1_checkpoint']}")
    print("Make sure to provide the correct path using --task1_checkpoint argument")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

# Train the selected fold with pretrained weights
print(f"\n{'='*60}")
print(f"TRAINING FOLD {SELECTED_FOLD} WITH PRETRAINED WEIGHTS")
print(f"{'='*60}")

train_cmd = [
    "nnUNetv2_train", 
    str(CONFIG['dataset_id']), 
    "3d_fullres", 
    str(SELECTED_FOLD), 
    "-p", CONFIG['plans_name'], 
    "-tr", CONFIG['trainer_name'],
    "-pretrained_weights", CONFIG['task1_checkpoint']
]

print(f"Command: {' '.join(train_cmd)}")
subprocess.run(train_cmd)

# Store fold results
results_dir = os.path.join(base_dir, "nnUNet_results", CONFIG['dataset_name'], 
                           f"{CONFIG['trainer_name']}__{CONFIG['plans_name']}__3d_fullres")
fold_dir = os.path.join(results_dir, f"fold_{SELECTED_FOLD}")

# Find best checkpoint
checkpoint_path = None
checkpoint_files = ["checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth"]
for ckpt in checkpoint_files:
    ckpt_path = os.path.join(fold_dir, ckpt)
    if os.path.exists(ckpt_path):
        checkpoint_path = ckpt_path
        print(f"Found checkpoint for fold {SELECTED_FOLD}: {ckpt}")
        break

# ============================================
# Evaluation Phase
# ============================================
print(f"\n{'='*60}")
print(f"EVALUATING FOLD {SELECTED_FOLD}")
print(f"{'='*60}")

val_cases = TASK2_SPLITS[SELECTED_FOLD]['val']
pred_output_dir = os.path.join(base_dir, f"evaluation_results_task2_fold{SELECTED_FOLD}", "predictions")
os.makedirs(pred_output_dir, exist_ok=True)

# Generate predictions
print("Generating predictions for validation set...")
print(f"Validation cases ({len(val_cases)}): {', '.join(val_cases)}")

success = generate_predictions_for_fold(
    fold=SELECTED_FOLD,
    dataset_name=CONFIG['dataset_name'],
    plans_name=CONFIG['plans_name'],
    trainer_name=CONFIG['trainer_name'],
    output_dir=pred_output_dir,
    val_cases=val_cases
)

if success:
    print("\nConverting 3-class predictions to binary and evaluating...")
    
    # Evaluate predictions (with 3-class to binary conversion)
    eval_results = evaluate_fold_predictions_with_conversion(
        pred_dir=pred_output_dir,
        gt_dir=os.path.join(CONFIG['base_dir'], "nnUNet_raw", CONFIG['dataset_name'], "labelsTr"),
        val_cases=val_cases,
        fold=SELECTED_FOLD
    )
    
    if eval_results:
        # Save fold evaluation results
        eval_save_path = os.path.join(base_dir, f"evaluation_results_task2_fold{SELECTED_FOLD}", "metrics.json")
        with open(eval_save_path, "w") as f:
            json.dump(eval_results, f, indent=4)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"FOLD {SELECTED_FOLD} EVALUATION RESULTS:")
        print(f"{'='*60}")
        print(f"  Mean Volumetric Dice: {eval_results['aggregates']['mean_volumetric_dice']:.4f}")
        print(f"  Mean Surface Dice (5mm): {eval_results['aggregates']['mean_surface_dice']:.4f}")
        print(f"  Mean Hausdorff95: {eval_results['aggregates']['mean_hausdorff95']:.2f} mm")
        print(f"  Mean MASD: {eval_results['aggregates']['mean_masd']:.2f} mm")
        print(f"  Tumor Burden RMSE: {eval_results['aggregates']['tumor_burden_rmse']:.2f} mm³")
        
        # Save fold summary
        fold_summary = {
            "fold": SELECTED_FOLD,
            "task": "Task 2 Fine-tuning",
            "approach": "Fine-tuning from Task 1 Fold 0 with minimal modifications",
            "pretrained_checkpoint": CONFIG['task1_checkpoint'],
            "training_samples": len(TASK2_SPLITS[SELECTED_FOLD]['train']),
            "training_cases": TASK2_SPLITS[SELECTED_FOLD]['train'],
            "validation_samples": len(val_cases),
            "validation_cases": val_cases,
            "num_epochs": CONFIG['num_epochs'],
            "learning_rate": CONFIG['initial_lr'],
            "model_checkpoint": checkpoint_path,
            "evaluation_metrics": eval_results['aggregates']
        }
        
        summary_path = os.path.join(base_dir, f"evaluation_results_task2_fold{SELECTED_FOLD}", "fold_summary.json")
        with open(summary_path, "w") as f:
            json.dump(fold_summary, f, indent=4)
        
        print(f"\nFold summary saved to: {summary_path}")
    else:
        print(f"ERROR: Evaluation failed for fold {SELECTED_FOLD}")
else:
    print(f"ERROR: Prediction generation failed for fold {SELECTED_FOLD}")

# ============================================
# Final Summary for This Fold
# ============================================
print("\n" + "="*80)
print(f"FOLD {SELECTED_FOLD} FINE-TUNING COMPLETED")
print("="*80)
print(f"\nSummary:")
print(f"  Task: Task 2 (MR-Linac T2-weighted MRIs)")
print(f"  Training samples: {len(TASK2_SPLITS[SELECTED_FOLD]['train'])}")
print(f"  Validation samples: {len(TASK2_SPLITS[SELECTED_FOLD]['val'])}")
print(f"  Number of epochs: {CONFIG['num_epochs']}")
print(f"  Learning rate: {CONFIG['initial_lr']}")
print(f"  Pretrained from: Task 1 Fold 0")

print(f"\nResults saved in:")
print(f"  - Model: {fold_dir}")
print(f"  - Evaluation: {os.path.join(base_dir, f'evaluation_results_task2_fold{SELECTED_FOLD}')}")

print("\nFine-tuning pipeline completed successfully!")
print("\nTo train other folds, run:")
for f in range(3):
    if f != SELECTED_FOLD:
        print(f"  python finetune_task2.py --fold {f}")

print("\n" + "="*80)
print("REQUIRED FOLDER STRUCTURE REMINDER")
print("="*80)
print("Make sure you have prepared these folders before running:")
print("1. ./data/PANTHER_Task2/ImagesTr/  (50 MHA files)")
print("2. ./data/PANTHER_Task2/LabelsTr/  (50 MHA files)")
print("3. Task 1 checkpoint at the specified path")
print("4. evaluate_local_fixed.py in current directory")
