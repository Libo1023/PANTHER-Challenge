import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import shutil
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import subprocess
import sys

# Import the FIXED evaluation function
sys.path.append('./')
from evaluate_local_fixed import evaluate_segmentation_performance

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    # Dataset configuration
    "dataset_id": 91,  # Using 91 for student dataset
    "dataset_name": "Dataset091_PantherStudent",
    
    # RESIDUAL ENCODER VARIANT SELECTION
    "resenc_variant": "M",  # Using ResEnc-M as specified
    
    # Training hyperparameters
    "num_epochs_student": 800,  # Quick test as requested
    "initial_lr": 0.005,  # Default nnU-Net learning rate
    
    # Training configuration
    "folds_to_train": [0, 1, 2, 3, 4],
    "trainer_name": "nnUNetTrainer_Student",  # Custom trainer for student
    
    # Base directory
    "base_dir": ".",
    
    # Data paths
    "labeled_images": "./data/PANTHER_Task1/ImagesTr/",
    "labeled_labels": "./data/PANTHER_Task1/LabelsTr/",
    "unlabeled_images": "./data/PANTHER_Task1/ImagesTr_unlabeled/",
    "pseudo_labels": "./data/PANTHER_Task1/PredictionsTr_unlabeled_3class/"
}

# ResEnc configurations
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

print(f"Noisy Student Training Pipeline (No Fine-tuning)")
print(f"ResEnc variant: {CONFIG['resenc_variant']}")
print(f"Expected VRAM usage: {CONFIG['expected_vram']}")
print(f"Student epochs: {CONFIG['num_epochs_student']}")
print(f"Training approach: All pseudo labels added to training set")

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
os.makedirs(os.path.join(base_dir, "evaluation_results_student"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "custom_trainers"), exist_ok=True)

print("\nEnvironment variables set:")
print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
print(f"nnUNet_results: {os.environ['nnUNet_results']}")

# ============================================
# Helper Functions
# ============================================
def check_pseudo_label_has_foreground(label_path):
    """Check if a pseudo label contains any foreground (non-background) voxels"""
    try:
        img = sitk.ReadImage(label_path)
        arr = sitk.GetArrayFromImage(img)
        # Check if there are any non-zero values (tumor=1 or pancreas=2)
        return np.any(arr > 0)
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return False

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

def evaluate_fold_predictions(pred_dir, gt_dir, val_cases, fold):
    """Evaluate predictions for a specific fold using evaluate_local_fixed.py"""
    
    # Create a temporary GT directory with only validation cases
    temp_gt_dir = os.path.join(CONFIG['base_dir'], f"temp_gt_fold_{fold}")
    if os.path.exists(temp_gt_dir):
        shutil.rmtree(temp_gt_dir)
    os.makedirs(temp_gt_dir, exist_ok=True)
    
    # Copy ground truth labels for validation cases
    raw_labels_dir = os.path.join(CONFIG['base_dir'], "nnUNet_raw", CONFIG['dataset_name'], "labelsTr")
    copied_count = 0
    for case in val_cases:
        src_file = os.path.join(raw_labels_dir, f"{case}.mha")
        if os.path.exists(src_file):
            shutil.copy2(src_file, temp_gt_dir)
            copied_count += 1
    
    print(f"Copied {copied_count} ground truth labels for evaluation")
    
    # Run evaluation
    try:
        results = evaluate_segmentation_performance(
            pred_dir=pred_dir,
            gt_dir=temp_gt_dir,
            subject_list=val_cases,
            verbose=True
        )
        
        # Clean up
        shutil.rmtree(temp_gt_dir)
        
        return results
    except Exception as e:
        print(f"ERROR evaluating fold {fold}: {e}")
        # Clean up
        if os.path.exists(temp_gt_dir):
            shutil.rmtree(temp_gt_dir)
        return None

# ============================================
# Create Custom Student Trainer
# ============================================

# Student trainer - minimal, just sets epochs
student_trainer_code = f'''
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class {CONFIG["trainer_name"]}(nnUNetTrainer):
    """Minimal trainer for student phase - only sets number of epochs"""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # Override number of epochs
        self.num_epochs = {CONFIG["num_epochs_student"]}
        self.initial_lr = {CONFIG["initial_lr"]}
        
    def on_train_epoch_start(self):
        """Log training progress"""
        super().on_train_epoch_start()
        if self.current_epoch % 1 == 0:
            self.print_to_log_file(f"Student Training - Epoch {{self.current_epoch}}/{{self.num_epochs}}")
'''

# Save custom trainer
trainer_path = os.path.join(base_dir, "custom_trainers", f"{CONFIG['trainer_name']}.py")
with open(trainer_path, 'w') as f:
    f.write(student_trainer_code)
print(f"\nStudent trainer created at: {trainer_path}")

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
# Filter Pseudo Labels
# ============================================
print("\n" + "="*60)
print("FILTERING PSEUDO LABELS")
print("="*60)

# Get all pseudo-labeled files
pseudo_label_files = list(Path(CONFIG["pseudo_labels"]).glob("*.mha"))
print(f"Total pseudo-labeled files found: {len(pseudo_label_files)}")

# Filter out background-only predictions
valid_pseudo_labels = []
background_only_count = 0

for label_file in pseudo_label_files:
    if check_pseudo_label_has_foreground(str(label_file)):
        valid_pseudo_labels.append(label_file)
    else:
        background_only_count += 1

print(f"Background-only predictions (excluded): {background_only_count}")
print(f"Valid pseudo-labeled samples: {len(valid_pseudo_labels)}")

# ============================================
# Create Dataset Structure
# ============================================
print("\n" + "="*60)
print("CREATING DATASET STRUCTURE")
print("="*60)

dataset_dir = os.path.join(base_dir, "nnUNet_raw", CONFIG['dataset_name'])
os.makedirs(os.path.join(dataset_dir, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "labelsTr"), exist_ok=True)

dest_images = os.path.join(dataset_dir, "imagesTr")
dest_labels = os.path.join(dataset_dir, "labelsTr")

# Keep track of case IDs
labeled_cases = []
pseudo_cases = []

# Copy labeled images and labels
print("\nCopying labeled data...")
labeled_count = 0
for img_file in Path(CONFIG["labeled_images"]).glob("*.mha"):
    case_id = img_file.stem.replace("_0001_0000", "")
    
    # Copy image
    dest_img_path = Path(dest_images) / f"{case_id}_0000.mha"
    shutil.copy2(img_file, dest_img_path)
    
    # Copy label
    label_file = Path(CONFIG["labeled_labels"]) / f"{case_id}_0001.mha"
    if label_file.exists():
        dest_label_path = Path(dest_labels) / f"{case_id}.mha"
        shutil.copy2(label_file, dest_label_path)
        
        labeled_cases.append(case_id)
        labeled_count += 1

print(f"Copied {labeled_count} labeled samples")

# Copy valid pseudo-labeled data
print("\nCopying pseudo-labeled data...")
pseudo_count = 0

# Create a mapping of unlabeled images for faster lookup
unlabeled_images_map = {}
for img_file in Path(CONFIG["unlabeled_images"]).glob("*.mha"):
    # Use first 10 characters as key
    key = img_file.stem[:10]
    unlabeled_images_map[key] = img_file

for pseudo_label in valid_pseudo_labels:
    # Extract case ID from pseudo label name
    pseudo_case_id = pseudo_label.stem
    
    # Find corresponding unlabeled image using first 10 characters
    key = pseudo_case_id[:10]
    found_image = unlabeled_images_map.get(key)
    
    if found_image:
        # Create a unique case ID for the pseudo-labeled sample
        unique_case_id = f"pseudo_{pseudo_case_id}"
        
        # Copy image
        dest_img_path = Path(dest_images) / f"{unique_case_id}_0000.mha"
        shutil.copy2(found_image, dest_img_path)
        
        # Copy pseudo label
        dest_label_path = Path(dest_labels) / f"{unique_case_id}.mha"
        shutil.copy2(pseudo_label, dest_label_path)
        
        pseudo_cases.append(unique_case_id)
        pseudo_count += 1

print(f"Copied {pseudo_count} pseudo-labeled samples")

# ============================================
# Create dataset.json
# ============================================
total_training = len(labeled_cases) + len(pseudo_cases)
print(f"\nTotal training cases: {total_training} ({len(labeled_cases)} labeled + {len(pseudo_cases)} pseudo)")

dataset_json = {
    "channel_names": {
        "0": "MRI"
    },
    "labels": {
        "background": 0,
        "tumor": 1,
        "pancreas": 2
    },
    "numTraining": total_training,
    "file_ending": ".mha"
}

dataset_json_path = os.path.join(dataset_dir, "dataset.json")
with open(dataset_json_path, "w") as f:
    json.dump(dataset_json, f, indent=4)

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
# Create Custom Splits Based on train_nnunet.py
# ============================================
print("\nCreating custom 5-fold splits...")

# These are the exact splits from train_nnunet.py
original_splits = [
    {
        "train": [
            "10014", "10100", "10053", "10034", "10037", "10113", "10058", "10023",
            "10044", "10072", "10095", "10106", "10112", "10047", "10027", "10108",
            "10080", "10051", "10049", "10048", "10001", "10110", "10046", "10075",
            "10028", "10042", "10078", "10085", "10103", "10025", "10060", "10016",
            "10118", "10059", "10123", "10026", "10054", "10050", "10115", "10099",
            "10012", "10018", "10117", "10041", "10093", "10105", "10120", "10015",
            "10116", "10030", "10029", "10090", "10088", "10039", "10061", "10083",
            "10133", "10131", "10129", "10132", "10092", "10033", "10021", "10121",
            "10019", "10087", "10007", "10000", "10091", "10084", "10094", "10006",
            "10063", "10077"
        ],
        "val": [
            "10011", "10040", "10055", "10032", "10068", "10057", "10031", "10089",
            "10002", "10066", "10070", "10125", "10067", "10038", "10017", "10074",
            "10102", "10128"
        ]
    },
    {
        "train": [
            "10011", "10014", "10100", "10053", "10034", "10037", "10113", "10058",
            "10023", "10044", "10072", "10095", "10106", "10112", "10047", "10027",
            "10108", "10080", "10051", "10049", "10048", "10001", "10110", "10046",
            "10075", "10028", "10042", "10040", "10055", "10032", "10068", "10057",
            "10031", "10089", "10002", "10066", "10070", "10125", "10067", "10038",
            "10017", "10074", "10102", "10128", "10092", "10033", "10021", "10121",
            "10019", "10087", "10007", "10000", "10091", "10084", "10094", "10006",
            "10063", "10077", "10103", "10025", "10060", "10016", "10054", "10115",
            "10099", "10012", "10018", "10117", "10041", "10093", "10105", "10120",
            "10015", "10116"
        ],
        "val": [
            "10078", "10085", "10118", "10059", "10123", "10026", "10050", "10030",
            "10029", "10090", "10088", "10039", "10061", "10083", "10133", "10131",
            "10129", "10132"
        ]
    },
    {
        "train": [
            "10011", "10014", "10100", "10053", "10034", "10037", "10113", "10058",
            "10023", "10044", "10072", "10095", "10106", "10112", "10047", "10027",
            "10108", "10080", "10051", "10049", "10078", "10085", "10118", "10059",
            "10123", "10026", "10050", "10030", "10029", "10090", "10088", "10039",
            "10061", "10083", "10133", "10131", "10129", "10132", "10040", "10055",
            "10032", "10068", "10057", "10031", "10089", "10002", "10066", "10070",
            "10125", "10067", "10038", "10017", "10074", "10102", "10128", "10092",
            "10033", "10021", "10121", "10019", "10087", "10007", "10000", "10091",
            "10084", "10094", "10006", "10063", "10077", "10025", "10054", "10115",
            "10099", "10018"
        ],
        "val": [
            "10048", "10001", "10110", "10046", "10075", "10028", "10042", "10103",
            "10060", "10016", "10012", "10117", "10041", "10093", "10105", "10120",
            "10015", "10116"
        ]
    },
    {
        "train": [
            "10011", "10048", "10001", "10110", "10046", "10075", "10028", "10042",
            "10103", "10060", "10016", "10012", "10117", "10041", "10093", "10105",
            "10120", "10015", "10116", "10078", "10085", "10118", "10059", "10123",
            "10026", "10050", "10030", "10029", "10090", "10088", "10039", "10061",
            "10083", "10133", "10131", "10129", "10132", "10040", "10055", "10032",
            "10068", "10057", "10031", "10089", "10002", "10066", "10070", "10125",
            "10067", "10038", "10017", "10074", "10102", "10128", "10092", "10033",
            "10021", "10121", "10019", "10087", "10007", "10000", "10091", "10084",
            "10094", "10006", "10063", "10077", "10103", "10025", "10060", "10016",
            "10054", "10050"
        ],
        "val": [
            "10014", "10100", "10053", "10034", "10037", "10113", "10058", "10023",
            "10044", "10072", "10095", "10106", "10112", "10047", "10027", "10108",
            "10080", "10051"
        ]
    },
    {
        "train": [
            "10011", "10014", "10100", "10053", "10034", "10037", "10113", "10058",
            "10023", "10044", "10072", "10095", "10106", "10112", "10047", "10027",
            "10108", "10080", "10051", "10048", "10001", "10110", "10046", "10075",
            "10028", "10042", "10103", "10060", "10016", "10012", "10117", "10041",
            "10093", "10105", "10120", "10015", "10116", "10078", "10085", "10118",
            "10059", "10123", "10026", "10050", "10030", "10029", "10090", "10088",
            "10039", "10061", "10083", "10133", "10131", "10129", "10132", "10040",
            "10055", "10032", "10068", "10057", "10031", "10089", "10002", "10066",
            "10070", "10125", "10067", "10038", "10017", "10074", "10102", "10128"
        ],
        "val": [
            "10049", "10092", "10033", "10021", "10121", "10019", "10087", "10007",
            "10000", "10091", "10084", "10094", "10006", "10063", "10077", "10025",
            "10054", "10115", "10099", "10018"
        ]
    }
]

# Now modify these splits to add ALL pseudo cases to training
modified_splits = []
for fold_idx, split in enumerate(original_splits):
    # Keep validation as is (only labeled cases)
    val_cases = split["val"]
    
    # Add all pseudo cases to training
    train_cases = split["train"] + pseudo_cases
    
    modified_splits.append({
        "train": train_cases,
        "val": val_cases
    })
    
    print(f"Fold {fold_idx}: {len(train_cases)} train ({len(split['train'])} labeled + {len(pseudo_cases)} pseudo), {len(val_cases)} val")

# Save splits
preprocessed_dataset_dir = os.path.join(base_dir, "nnUNet_preprocessed", CONFIG['dataset_name'])
splits_path = os.path.join(preprocessed_dataset_dir, "splits_final.json")
with open(splits_path, "w") as f:
    json.dump(modified_splits, f, indent=2)

print(f"Saved modified splits to: {splits_path}")

# ============================================
# Student Training Phase with Evaluation
# ============================================
print("\n" + "="*80)
print("NOISY STUDENT TRAINING (5 FOLDS)")
print("="*80)
print(f"Configuration:")
print(f"  Architecture: ResEnc{CONFIG['resenc_variant']}")
print(f"  Epochs: {CONFIG['num_epochs_student']}")
print(f"  Learning Rate: {CONFIG['initial_lr']}")
print(f"  Training approach: All pseudo labels added to each fold")
print("="*80)

fold_results = {}
fold_metrics = {}
student_checkpoints = {}

for fold in CONFIG["folds_to_train"]:
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold}")
    print(f"{'='*60}")
    
    # Train
    train_cmd = [
        "nnUNetv2_train", 
        str(CONFIG['dataset_id']), 
        "3d_fullres", 
        str(fold), 
        "-p", CONFIG['plans_name'], 
        "-tr", CONFIG['trainer_name']
    ]
    
    print(f"Command: {' '.join(train_cmd)}")
    subprocess.run(train_cmd)
    
    # Store fold location
    results_dir = os.path.join(base_dir, "nnUNet_results", CONFIG['dataset_name'], 
                               f"{CONFIG['trainer_name']}__{CONFIG['plans_name']}__3d_fullres")
    fold_dir = os.path.join(results_dir, f"fold_{fold}")
    fold_results[fold] = fold_dir
    
    # Find and store best checkpoint
    checkpoint_files = ["checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth"]
    for ckpt in checkpoint_files:
        ckpt_path = os.path.join(fold_dir, ckpt)
        if os.path.exists(ckpt_path):
            student_checkpoints[fold] = ckpt_path
            print(f"Found checkpoint for fold {fold}: {ckpt}")
            break
    
    # Generate predictions for validation set
    print(f"\n{'='*40}")
    print(f"EVALUATING FOLD {fold}")
    print(f"{'='*40}")
    
    val_cases = modified_splits[fold]['val']
    pred_output_dir = os.path.join(base_dir, "evaluation_results_student", f"fold_{fold}_predictions")
    os.makedirs(pred_output_dir, exist_ok=True)
    
    # Generate predictions
    success = generate_predictions_for_fold(
        fold=fold,
        dataset_name=CONFIG['dataset_name'],
        plans_name=CONFIG['plans_name'],
        trainer_name=CONFIG['trainer_name'],
        output_dir=pred_output_dir,
        val_cases=val_cases
    )
    
    if success:
        # Evaluate predictions
        eval_results = evaluate_fold_predictions(
            pred_dir=pred_output_dir,
            gt_dir=os.path.join(CONFIG['base_dir'], "nnUNet_raw", CONFIG['dataset_name'], "labelsTr"),
            val_cases=val_cases,
            fold=fold
        )
        
        if eval_results:
            fold_metrics[fold] = eval_results
            
            # Save fold evaluation results
            eval_save_path = os.path.join(base_dir, "evaluation_results_student", f"fold_{fold}_metrics.json")
            with open(eval_save_path, "w") as f:
                json.dump(eval_results, f, indent=4)
            
            # Print summary
            print(f"\nFOLD {fold} EVALUATION RESULTS:")
            print(f"  Mean Volumetric Dice: {eval_results['aggregates']['mean_volumetric_dice']:.4f}")
            print(f"  Mean Surface Dice (5mm): {eval_results['aggregates']['mean_surface_dice']:.4f}")
            print(f"  Mean Hausdorff95: {eval_results['aggregates']['mean_hausdorff95']:.2f} mm")
            print(f"  Mean MASD: {eval_results['aggregates']['mean_masd']:.2f} mm")
            print(f"  Tumor Burden RMSE: {eval_results['aggregates']['tumor_burden_rmse']:.2f} mm³")
        else:
            print(f"ERROR: Evaluation failed for fold {fold}")
    else:
        print(f"ERROR: Prediction generation failed for fold {fold}")

# ============================================
# Final Summary
# ============================================
print("\n" + "="*80)
print("STUDENT TRAINING SUMMARY")
print("="*80)

if fold_metrics:
    print(f"\nEvaluation Summary for All Student Models:")
    print(f"{'Fold':<6} {'Vol Dice':<10} {'Surf Dice':<12} {'HD95':<10} {'MASD':<10} {'RMSE Vol':<12}")
    print("-" * 70)
    
    for fold in sorted(fold_metrics.keys()):
        agg = fold_metrics[fold]['aggregates']
        print(f"{fold:<6} {agg['mean_volumetric_dice']:<10.4f} {agg['mean_surface_dice']:<12.4f} "
              f"{agg['mean_hausdorff95']:<10.2f} {agg['mean_masd']:<10.2f} {agg['tumor_burden_rmse']:<12.2f}")
    
    # Calculate average metrics across all folds
    avg_metrics = {
        "mean_volumetric_dice": np.mean([fold_metrics[f]['aggregates']['mean_volumetric_dice'] for f in fold_metrics]),
        "mean_surface_dice": np.mean([fold_metrics[f]['aggregates']['mean_surface_dice'] for f in fold_metrics]),
        "mean_hausdorff95": np.mean([fold_metrics[f]['aggregates']['mean_hausdorff95'] for f in fold_metrics]),
        "mean_masd": np.mean([fold_metrics[f]['aggregates']['mean_masd'] for f in fold_metrics]),
        "tumor_burden_rmse": np.mean([fold_metrics[f]['aggregates']['tumor_burden_rmse'] for f in fold_metrics])
    }
    
    print(f"\nAverage across all folds:")
    print(f"  Mean Volumetric Dice: {avg_metrics['mean_volumetric_dice']:.4f}")
    print(f"  Mean Surface Dice (5mm): {avg_metrics['mean_surface_dice']:.4f}")
    print(f"  Mean Hausdorff95: {avg_metrics['mean_hausdorff95']:.2f} mm")
    print(f"  Mean MASD: {avg_metrics['mean_masd']:.2f} mm")
    print(f"  Tumor Burden RMSE: {avg_metrics['tumor_burden_rmse']:.2f} mm³")

print("\n" + "="*80)
print("NOISY STUDENT TRAINING COMPLETED")
print("="*80)
print(f"\nSummary:")
print(f"  Total training samples per fold: {len(labeled_cases) + len(pseudo_cases)}")
print(f"  - Labeled: {len(labeled_cases)}")
print(f"  - Pseudo-labeled: {len(pseudo_cases)}")
print(f"  - Excluded (background-only): {background_only_count}")
print(f"  Number of epochs: {CONFIG['num_epochs_student']}")
print(f"  Learning rate: {CONFIG['initial_lr']}")

print(f"\nResults saved in:")
print(f"  - 5 Student models: {os.path.join(base_dir, 'nnUNet_results', CONFIG['dataset_name'])}")
print(f"  - Evaluation results: {os.path.join(base_dir, 'evaluation_results_student')}")

# Save final summary
final_summary = {
    "approach": "Noisy student training without fine-tuning",
    "num_models": 5,
    "labeled_samples": len(labeled_cases),
    "pseudo_samples": len(pseudo_cases),
    "excluded_pseudo": background_only_count,
    "num_epochs": CONFIG['num_epochs_student'],
    "learning_rate": CONFIG['initial_lr'],
    "student_models": fold_results,
    "evaluation_metrics": {f: fold_metrics[f]['aggregates'] for f in fold_metrics} if fold_metrics else {},
    "average_metrics": avg_metrics if 'avg_metrics' in locals() else None,
    "per_fold_checkpoints": student_checkpoints
}

summary_path = os.path.join(base_dir, "evaluation_results_student", "final_summary.json")
with open(summary_path, "w") as f:
    json.dump(final_summary, f, indent=4)

print(f"\nFinal summary saved to: {summary_path}")
print("\nTraining pipeline completed successfully!")
