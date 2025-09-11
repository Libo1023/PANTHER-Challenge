import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import shutil
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import subprocess
import sys
import random
import time
from skimage.exposure import match_histograms

# Import the FIXED evaluation function
sys.path.append('./')
from evaluate_local_fixed import evaluate_segmentation_performance

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    # Dataset configuration for combined dataset
    "dataset_id": 93,
    "dataset_name": "Dataset093_PANTHERCombined",
    
    # RESIDUAL ENCODER VARIANT SELECTION
    "resenc_variant": "M",  # Using ResEnc-M
    
    # Training hyperparameters
    "num_epochs": 800, 
    "initial_lr": 0.0005,
    
    # Trainer name
    "trainer_name": "nnUNetTrainer_Combined",
    
    # Base directory
    "base_dir": ".",
    
    # Task 1 data paths
    "task1_labeled_images": "./data/PANTHER_Task1/ImagesTr/",
    "task1_labeled_labels": "./data/PANTHER_Task1/LabelsTr/",
    "task1_unlabeled_images": "./data/PANTHER_Task1/ImagesTr_unlabeled/",
    "task1_pseudo_labels": "./data/PANTHER_Task1/PredictionsTr_unlabeled_3class/",
    
    # Task 2 data paths
    "task2_images": "./data/PANTHER_Task2/ImagesTr/",
    "task2_labels": "./data/PANTHER_Task2/LabelsTr/",
    
    # Number of Task 2 validation samples
    "num_task2_val": 4,
    
    # Domain adaptation method
    "use_domain_adaptation": True,
    "adaptation_method": "histogram_matching"  # Options: "histogram_matching", "z_score", "none"
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

print("="*80)
print("COMBINED TASK 1 AND TASK 2 TRAINING PIPELINE WITH DOMAIN ADAPTATION")
print("="*80)
print(f"ResEnc variant: {CONFIG['resenc_variant']}")
print(f"Expected VRAM usage: {CONFIG['expected_vram']}")
print(f"Training epochs: {CONFIG['num_epochs']}")
print(f"Initial learning rate: {CONFIG['initial_lr']}")
print(f"Dataset: {CONFIG['dataset_name']}")
print(f"Domain Adaptation: {CONFIG['use_domain_adaptation']}")
print(f"Adaptation Method: {CONFIG['adaptation_method']}")

if CONFIG["use_domain_adaptation"]:
    print("\n" + "="*60)
    print("PIPELINE OVERVIEW")
    print("="*60)
    print("1. Discover and count all data files")
    print("2. Filter pseudo-labels (exclude background-only)")
    print("3. Compute Task 2 intensity statistics")
    print("4. Transform Task 1 images to Task 2 domain")
    print("5. Create combined dataset")
    print("6. Train model from scratch")
    print("7. Evaluate on Task 2 validation set")
    print("="*60)

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
os.makedirs(os.path.join(base_dir, "evaluation_results_combined"), exist_ok=True)
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

def compute_task2_statistics(task2_image_paths):
    """Compute intensity statistics from all Task 2 images"""
    print("\nComputing Task 2 intensity statistics...")
    
    all_intensities = []
    
    for img_path in task2_image_paths:
        img = sitk.ReadImage(str(img_path))
        arr = sitk.GetArrayFromImage(img)
        all_intensities.extend(arr.flatten())
    
    all_intensities = np.array(all_intensities)
    
    stats = {
        'mean': np.mean(all_intensities),
        'std': np.std(all_intensities),
        'min': np.min(all_intensities),
        'max': np.max(all_intensities),
        'percentiles': np.percentile(all_intensities, [1, 5, 25, 50, 75, 95, 99]),
        'histogram': all_intensities  # Keep for histogram matching
    }
    
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Min: {stats['min']:.2f}")
    print(f"  Max: {stats['max']:.2f}")
    print(f"  Median: {stats['percentiles'][3]:.2f}")
    
    return stats

def apply_histogram_matching(source_image_path, target_stats):
    """Apply histogram matching to transform source image to target domain"""
    # Read source image
    source_img = sitk.ReadImage(str(source_image_path))
    source_arr = sitk.GetArrayFromImage(source_img)
    
    # Store original dtype and shape
    original_dtype = source_arr.dtype
    original_shape = source_arr.shape
    
    # Sample from target histogram for matching
    target_sample = target_stats['histogram']
    if len(target_sample) > 1000000:  # If more than 1M voxels, sample
        np.random.seed(42)  # For reproducibility
        idx = np.random.choice(len(target_sample), 1000000, replace=False)
        target_sample = target_sample[idx]
    
    # Convert to float for processing
    source_flat = source_arr.flatten().astype(np.float32)
    target_sample = target_sample.astype(np.float32)
    
    # Apply histogram matching using skimage
    # Note: match_histograms works with float arrays
    matched_flat = match_histograms(source_flat.reshape(-1, 1), 
                                   target_sample.reshape(-1, 1), 
                                   channel_axis=None)
    
    # Reshape back to original shape
    matched_arr = matched_flat.reshape(original_shape)
    
    # Convert back to original dtype
    if original_dtype in [np.int16, np.int32, np.int64]:
        matched_arr = np.round(matched_arr).astype(original_dtype)
    else:
        matched_arr = matched_arr.astype(original_dtype)
    
    # Create new image with matched intensities
    matched_img = sitk.GetImageFromArray(matched_arr)
    matched_img.CopyInformation(source_img)
    
    return matched_img

def apply_z_score_matching(source_image_path, target_stats):
    """Apply z-score normalization to match target statistics"""
    # Read source image
    source_img = sitk.ReadImage(str(source_image_path))
    source_arr = sitk.GetArrayFromImage(source_img)
    
    # Store original dtype
    original_dtype = source_arr.dtype
    
    # Convert to float for processing
    source_arr = source_arr.astype(np.float32)
    
    # Compute source statistics
    source_mean = np.mean(source_arr)
    source_std = np.std(source_arr)
    
    # Apply z-score transformation
    if source_std > 0:
        normalized_arr = (source_arr - source_mean) / source_std
        matched_arr = normalized_arr * target_stats['std'] + target_stats['mean']
    else:
        matched_arr = source_arr
    
    # Convert back to original dtype
    if original_dtype in [np.int16, np.int32, np.int64]:
        matched_arr = np.round(matched_arr).astype(original_dtype)
    else:
        matched_arr = matched_arr.astype(original_dtype)
    
    # Create new image
    matched_img = sitk.GetImageFromArray(matched_arr)
    matched_img.CopyInformation(source_img)
    
    return matched_img

def transform_image_to_target_domain(source_image_path, target_stats, method="histogram_matching"):
    """Transform source image to match target domain statistics"""
    if method == "histogram_matching":
        return apply_histogram_matching(source_image_path, target_stats)
    elif method == "z_score":
        return apply_z_score_matching(source_image_path, target_stats)
    else:
        # No transformation
        return sitk.ReadImage(str(source_image_path))

def generate_predictions(dataset_name, plans_name, trainer_name, output_dir, val_cases):
    """Generate predictions for validation cases"""
    
    # Create temporary directory for validation images
    temp_val_dir = os.path.join(CONFIG['base_dir'], "temp_val_combined")
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
    
    # Run prediction (fold 0 since we're not using cross-validation)
    pred_cmd = [
        "nnUNetv2_predict",
        "-i", temp_val_dir,
        "-o", output_dir,
        "-d", str(CONFIG['dataset_id']),
        "-p", plans_name,
        "-c", "3d_fullres",
        "-f", "0",
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

def evaluate_predictions_with_conversion(pred_dir, gt_dir, val_cases):
    """Evaluate predictions after converting 3-class to binary"""
    
    # Create directories for binary predictions and ground truth
    binary_pred_dir = os.path.join(CONFIG['base_dir'], "temp_binary_pred_combined")
    temp_gt_dir = os.path.join(CONFIG['base_dir'], "temp_gt_combined")
    
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
        print(f"ERROR evaluating predictions: {e}")
        # Clean up
        if os.path.exists(binary_pred_dir):
            shutil.rmtree(binary_pred_dir)
        if os.path.exists(temp_gt_dir):
            shutil.rmtree(temp_gt_dir)
        return None

# ============================================
# Discover and Count All Data
# ============================================
print("\n" + "="*60)
print("DISCOVERING ALL DATA FILES")
print("="*60)

# Count Task 1 labeled data
task1_labeled_images = list(Path(CONFIG["task1_labeled_images"]).glob("*.mha")) if Path(CONFIG["task1_labeled_images"]).exists() else []
task1_labeled_labels = list(Path(CONFIG["task1_labeled_labels"]).glob("*.mha")) if Path(CONFIG["task1_labeled_labels"]).exists() else []
print(f"Task 1 labeled images found: {len(task1_labeled_images)}")
print(f"Task 1 labeled labels found: {len(task1_labeled_labels)}")

# Count Task 1 unlabeled/pseudo-labeled data
task1_unlabeled_images = list(Path(CONFIG["task1_unlabeled_images"]).glob("*.mha")) if Path(CONFIG["task1_unlabeled_images"]).exists() else []
task1_pseudo_labels = list(Path(CONFIG["task1_pseudo_labels"]).glob("*.mha")) if Path(CONFIG["task1_pseudo_labels"]).exists() else []
print(f"Task 1 unlabeled images found: {len(task1_unlabeled_images)}")
print(f"Task 1 pseudo labels found: {len(task1_pseudo_labels)}")

# Count Task 2 data
task2_images = list(Path(CONFIG["task2_images"]).glob("*.mha")) if Path(CONFIG["task2_images"]).exists() else []
task2_labels = list(Path(CONFIG["task2_labels"]).glob("*.mha")) if Path(CONFIG["task2_labels"]).exists() else []
print(f"Task 2 images found: {len(task2_images)}")
print(f"Task 2 labels found: {len(task2_labels)}")

# ============================================
# Filter Pseudo Labels (exclude background-only)
# ============================================
print("\n" + "="*60)
print("FILTERING PSEUDO LABELS")
print("="*60)

valid_pseudo_labels = []
background_only_count = 0

for label_file in task1_pseudo_labels:
    if check_pseudo_label_has_foreground(str(label_file)):
        valid_pseudo_labels.append(label_file)
    else:
        background_only_count += 1

print(f"Valid pseudo-labeled samples: {len(valid_pseudo_labels)}")
print(f"Background-only predictions (excluded): {background_only_count}")

# ============================================
# Compute Task 2 Statistics for Domain Adaptation
# ============================================
target_stats = None
if CONFIG["use_domain_adaptation"] and CONFIG["adaptation_method"] != "none":
    print("\n" + "="*60)
    print("COMPUTING TARGET DOMAIN STATISTICS")
    print("="*60)
    
    # Use all 50 Task 2 images for computing statistics
    target_stats = compute_task2_statistics(task2_images)
    print(f"Computed statistics from {len(task2_images)} Task 2 images")

# ============================================
# Select Task 2 Validation Samples
# ============================================
print("\n" + "="*60)
print("SELECTING VALIDATION SAMPLES FROM TASK 2")
print("="*60)

# Get all Task 2 case IDs
task2_case_ids = []
for img_file in task2_images:
    # Extract case ID (remove _0000.mha suffix if present)
    case_id = img_file.stem.replace("_0000", "")
    task2_case_ids.append(case_id)

# Sort case IDs for consistency
task2_case_ids = sorted(list(set(task2_case_ids)))
print(f"Total Task 2 cases: {len(task2_case_ids)}")

# Select last 4 samples for validation
val_cases = task2_case_ids[-CONFIG["num_task2_val"]:]
train_task2_cases = task2_case_ids[:-CONFIG["num_task2_val"]]

print(f"Task 2 validation samples ({len(val_cases)}): {', '.join(val_cases)}")
print(f"Task 2 training samples ({len(train_task2_cases)}): {len(train_task2_cases)} cases")

# Start timing the dataset creation
dataset_creation_start = time.time()

# ============================================
# Create Dataset Structure
# ============================================
print("\n" + "="*60)
print("CREATING COMBINED DATASET STRUCTURE")
print("="*60)

# Summary of transformation work
if CONFIG["use_domain_adaptation"] and target_stats is not None:
    total_to_transform = len(task1_labeled_images) + len(valid_pseudo_labels)
    print(f"\nDomain Adaptation Summary:")
    print(f"  Method: {CONFIG['adaptation_method']}")
    print(f"  Task 1 labeled images to transform: {len(task1_labeled_images)}")
    print(f"  Task 1 pseudo-labeled images to transform: {len(valid_pseudo_labels)}")
    print(f"  Total images to transform: {total_to_transform}")
    print(f"  Task 2 images (no transformation needed): {len(task2_images)}")
    estimated_time = total_to_transform * 2.5 / 60  # Assume ~2.5 seconds per image
    print(f"  Estimated transformation time: {estimated_time:.1f} minutes")
    print("")

dataset_dir = os.path.join(base_dir, "nnUNet_raw", CONFIG['dataset_name'])
dataset_json_path = os.path.join(dataset_dir, "dataset.json")

# Create dataset directories
os.makedirs(os.path.join(dataset_dir, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "labelsTr"), exist_ok=True)

dest_images = os.path.join(dataset_dir, "imagesTr")
dest_labels = os.path.join(dataset_dir, "labelsTr")

# Keep track of all cases
all_training_cases = []
all_validation_cases = val_cases.copy()

# ============================================
# Copy Task 1 Labeled Data with Domain Adaptation
# ============================================
print("\nCopying Task 1 labeled data...")
if CONFIG["use_domain_adaptation"] and target_stats is not None:
    print(f"  Applying {CONFIG['adaptation_method']} domain adaptation...")
    print(f"  Total images to transform: {len(task1_labeled_images)}")

task1_labeled_count = 0
transform_start_time = time.time()

for idx, img_file in enumerate(task1_labeled_images, 1):
    # Extract case ID
    case_id = img_file.stem.replace("_0001_0000", "").replace("_0000", "")
    
    # Apply domain adaptation if enabled
    if CONFIG["use_domain_adaptation"] and target_stats is not None:
        # Show progress
        print(f"  [{idx}/{len(task1_labeled_images)}] Transforming: {case_id}...", end="")
        img_start_time = time.time()
        
        transformed_img = transform_image_to_target_domain(
            img_file, 
            target_stats, 
            CONFIG["adaptation_method"]
        )
        # Save transformed image
        dest_img_path = Path(dest_images) / f"{case_id}_0000.mha"
        sitk.WriteImage(transformed_img, str(dest_img_path))
        
        img_time = time.time() - img_start_time
        print(f" Done ({img_time:.2f}s)")
        
        # Show estimated time remaining every 10 images
        if idx % 10 == 0:
            elapsed = time.time() - transform_start_time
            avg_time_per_img = elapsed / idx
            remaining = (len(task1_labeled_images) - idx) * avg_time_per_img
            print(f"    Progress: {idx}/{len(task1_labeled_images)} ({100*idx/len(task1_labeled_images):.1f}%) - Est. remaining: {remaining/60:.1f} min")
    else:
        # Copy image without transformation
        dest_img_path = Path(dest_images) / f"{case_id}_0000.mha"
        shutil.copy2(img_file, dest_img_path)
        if idx % 10 == 0:
            print(f"  [{idx}/{len(task1_labeled_images)}] Copied: {case_id}")
    
    # Find corresponding label (labels are not transformed)
    label_patterns = [
        f"{case_id}_0001.mha",
        f"{case_id}.mha"
    ]
    
    label_copied = False
    for pattern in label_patterns:
        label_file = Path(CONFIG["task1_labeled_labels"]) / pattern
        if label_file.exists():
            dest_label_path = Path(dest_labels) / f"{case_id}.mha"
            shutil.copy2(label_file, dest_label_path)
            all_training_cases.append(case_id)
            task1_labeled_count += 1
            label_copied = True
            break
    
    if not label_copied:
        print(f"  Warning: No label found for {case_id}")

if CONFIG["use_domain_adaptation"] and target_stats is not None:
    total_time = time.time() - transform_start_time
    print(f"\nCompleted {task1_labeled_count} Task 1 labeled samples")
    print(f"  Total transformation time: {total_time/60:.1f} minutes")
    print(f"  Average time per image: {total_time/task1_labeled_count:.2f} seconds")
    print(f"  Domain adaptation applied: {CONFIG['adaptation_method']}")
else:
    print(f"Copied {task1_labeled_count} Task 1 labeled samples")

# ============================================
# Copy Task 1 Pseudo-labeled Data with Domain Adaptation
# ============================================
print("\nCopying Task 1 pseudo-labeled data...")
if CONFIG["use_domain_adaptation"] and target_stats is not None:
    print(f"  Applying {CONFIG['adaptation_method']} domain adaptation...")
    print(f"  Total images to transform: {len(valid_pseudo_labels)}")

task1_pseudo_count = 0
transform_start_time = time.time()

# Create mapping of unlabeled images for faster lookup
unlabeled_images_map = {}
for img_file in task1_unlabeled_images:
    # Use first 10 characters as key
    key = img_file.stem[:10]
    unlabeled_images_map[key] = img_file

for idx, pseudo_label in enumerate(valid_pseudo_labels, 1):
    # Extract case ID from pseudo label name
    pseudo_case_id = pseudo_label.stem
    
    # Find corresponding unlabeled image
    key = pseudo_case_id[:10]
    found_image = unlabeled_images_map.get(key)
    
    if found_image:
        # Create unique case ID for pseudo-labeled sample
        unique_case_id = f"pseudo_{pseudo_case_id}"
        
        # Apply domain adaptation if enabled
        if CONFIG["use_domain_adaptation"] and target_stats is not None:
            # Show progress
            print(f"  [{idx}/{len(valid_pseudo_labels)}] Transforming: {unique_case_id[:30]}...", end="")
            img_start_time = time.time()
            
            transformed_img = transform_image_to_target_domain(
                found_image, 
                target_stats, 
                CONFIG["adaptation_method"]
            )
            # Save transformed image
            dest_img_path = Path(dest_images) / f"{unique_case_id}_0000.mha"
            sitk.WriteImage(transformed_img, str(dest_img_path))
            
            img_time = time.time() - img_start_time
            print(f" Done ({img_time:.2f}s)")
            
            # Show estimated time remaining every 20 images
            if idx % 20 == 0:
                elapsed = time.time() - transform_start_time
                avg_time_per_img = elapsed / idx
                remaining = (len(valid_pseudo_labels) - idx) * avg_time_per_img
                print(f"    Progress: {idx}/{len(valid_pseudo_labels)} ({100*idx/len(valid_pseudo_labels):.1f}%) - Est. remaining: {remaining/60:.1f} min")
        else:
            # Copy image without transformation
            dest_img_path = Path(dest_images) / f"{unique_case_id}_0000.mha"
            shutil.copy2(found_image, dest_img_path)
            if idx % 20 == 0:
                print(f"  [{idx}/{len(valid_pseudo_labels)}] Copied: {unique_case_id[:30]}")
        
        # Copy pseudo label (labels are not transformed)
        dest_label_path = Path(dest_labels) / f"{unique_case_id}.mha"
        shutil.copy2(pseudo_label, dest_label_path)
        
        all_training_cases.append(unique_case_id)
        task1_pseudo_count += 1

if CONFIG["use_domain_adaptation"] and target_stats is not None:
    total_time = time.time() - transform_start_time
    print(f"\nCompleted {task1_pseudo_count} Task 1 pseudo-labeled samples")
    print(f"  Total transformation time: {total_time/60:.1f} minutes")
    print(f"  Average time per image: {total_time/task1_pseudo_count:.2f} seconds")
    print(f"  Domain adaptation applied: {CONFIG['adaptation_method']}")
else:
    print(f"Copied {task1_pseudo_count} Task 1 pseudo-labeled samples")

# ============================================
# Copy Task 2 Data (No transformation needed)
# ============================================
print("\nCopying Task 2 data (no transformation - already in target domain)...")
task2_train_count = 0
task2_val_count = 0

for img_file in task2_images:
    # Extract case ID
    case_id = img_file.stem.replace("_0000", "")
    
    # Check if this is training or validation
    if case_id in train_task2_cases:
        # Copy to training set
        dest_img_path = Path(dest_images) / f"{case_id}_0000.mha"
        shutil.copy2(img_file, dest_img_path)
        
        # Find corresponding label
        label_file = Path(CONFIG["task2_labels"]) / f"{case_id}.mha"
        if label_file.exists():
            dest_label_path = Path(dest_labels) / f"{case_id}.mha"
            shutil.copy2(label_file, dest_label_path)
            all_training_cases.append(case_id)
            task2_train_count += 1
    
    elif case_id in val_cases:
        # Also copy validation samples (needed for nnUNet structure)
        dest_img_path = Path(dest_images) / f"{case_id}_0000.mha"
        shutil.copy2(img_file, dest_img_path)
        
        # Find corresponding label
        label_file = Path(CONFIG["task2_labels"]) / f"{case_id}.mha"
        if label_file.exists():
            dest_label_path = Path(dest_labels) / f"{case_id}.mha"
            shutil.copy2(label_file, dest_label_path)
            task2_val_count += 1

print(f"Copied {task2_train_count} Task 2 training samples")
print(f"Copied {task2_val_count} Task 2 validation samples")

# Summary of all data processing
if CONFIG["use_domain_adaptation"] and target_stats is not None:
    print("\n" + "="*60)
    print("DOMAIN ADAPTATION COMPLETE")
    print("="*60)
    print(f"Successfully transformed {task1_labeled_count + task1_pseudo_count} Task 1 images")
    print(f"Task 2 images copied without transformation: {task2_train_count + task2_val_count}")
    print("All images are now in the Task 2 domain for training")

# ============================================
# Create dataset.json
# ============================================
total_samples = len(all_training_cases) + len(all_validation_cases)
print(f"\nTotal dataset samples: {total_samples}")
print(f"  Training: {len(all_training_cases)}")
print(f"    - Task 1 labeled: {task1_labeled_count}")
print(f"    - Task 1 pseudo: {task1_pseudo_count}")
print(f"    - Task 2: {task2_train_count}")
print(f"  Validation: {len(all_validation_cases)} (Task 2 only)")

# Dataset creation timing summary
dataset_creation_time = time.time() - dataset_creation_start
print(f"\nDataset creation completed in {dataset_creation_time/60:.1f} minutes")

dataset_json = {
    "channel_names": {
        "0": "MRI"
    },
    "labels": {
        "background": 0,
        "tumor": 1,
        "pancreas": 2
    },
    "numTraining": total_samples,
    "file_ending": ".mha",
    "description": "Combined PANTHER Task 1 and Task 2 dataset with domain adaptation"
}

with open(dataset_json_path, "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"Created dataset.json at: {dataset_json_path}")

# ============================================
# Create Custom Trainer
# ============================================
print("\n" + "="*60)
print("CREATING CUSTOM TRAINER")
print("="*60)

trainer_code = f'''
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class {CONFIG["trainer_name"]}(nnUNetTrainer):
    """Custom trainer for combined Task 1 and Task 2 training"""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device=None):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # Override number of epochs and learning rate
        self.num_epochs = {CONFIG["num_epochs"]}
        self.initial_lr = {CONFIG["initial_lr"]}
        
    def on_train_epoch_start(self):
        """Log training progress"""
        super().on_train_epoch_start()
        if self.current_epoch % 50 == 0:
            self.print_to_log_file(f"Combined Training - Epoch {{self.current_epoch}}/{{self.num_epochs}}")
'''

# Save custom trainer
trainer_path = os.path.join(base_dir, "custom_trainers", f"{CONFIG['trainer_name']}.py")
with open(trainer_path, 'w') as f:
    f.write(trainer_code)
print(f"Custom trainer created at: {trainer_path}")

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
# Create Custom Splits (Single Fold)
# ============================================
print("\n" + "="*60)
print("CREATING CUSTOM SPLITS")
print("="*60)

# Create single fold split
splits = [
    {
        "train": all_training_cases,
        "val": all_validation_cases
    }
]

preprocessed_dataset_dir = os.path.join(base_dir, "nnUNet_preprocessed", CONFIG['dataset_name'])
os.makedirs(preprocessed_dataset_dir, exist_ok=True)
splits_path = os.path.join(preprocessed_dataset_dir, "splits_final.json")

with open(splits_path, "w") as f:
    json.dump(splits, f, indent=2)

print(f"Saved splits to: {splits_path}")
print(f"Training samples: {len(all_training_cases)}")
print(f"Validation samples: {len(all_validation_cases)}")

# Save dataset info
dataset_info = {
    "task1_labeled_count": task1_labeled_count,
    "task1_pseudo_count": task1_pseudo_count,
    "task1_background_only_count": background_only_count,
    "task2_train_count": task2_train_count,
    "task2_val_count": task2_val_count,
    "validation_cases": all_validation_cases,
    "total_training": len(all_training_cases),
    "total_validation": len(all_validation_cases),
    "domain_adaptation": {
        "enabled": CONFIG["use_domain_adaptation"],
        "method": CONFIG["adaptation_method"],
        "target_stats_from": f"All {len(task2_images)} Task 2 images"
    }
}

info_path = os.path.join(dataset_dir, "dataset_info.json")
with open(info_path, "w") as f:
    json.dump(dataset_info, f, indent=4)

# ============================================
# Training Phase
# ============================================
print("\n" + "="*80)
print("STARTING COMBINED TRAINING WITH DOMAIN ADAPTATION")
print("="*80)
print(f"Configuration:")
print(f"  Architecture: ResEnc{CONFIG['resenc_variant']}")
print(f"  Epochs: {CONFIG['num_epochs']}")
print(f"  Learning Rate: {CONFIG['initial_lr']}")
print(f"  Training samples: {len(all_training_cases)}")
print(f"  Validation samples: {len(all_validation_cases)}")
print(f"  Domain Adaptation: {CONFIG['use_domain_adaptation']}")
if CONFIG["use_domain_adaptation"]:
    print(f"  Adaptation Method: {CONFIG['adaptation_method']}")
print("="*80)

# Train (fold 0 since we're using single fold)
train_cmd = [
    "nnUNetv2_train", 
    str(CONFIG['dataset_id']), 
    "3d_fullres", 
    "0",  # Fold 0
    "-p", CONFIG['plans_name'], 
    "-tr", CONFIG['trainer_name']
]

print(f"Command: {' '.join(train_cmd)}")
subprocess.run(train_cmd)

# Store results
results_dir = os.path.join(base_dir, "nnUNet_results", CONFIG['dataset_name'], 
                           f"{CONFIG['trainer_name']}__{CONFIG['plans_name']}__3d_fullres")
fold_dir = os.path.join(results_dir, "fold_0")

# Find best checkpoint
checkpoint_path = None
checkpoint_files = ["checkpoint_best.pth", "checkpoint_final.pth", "checkpoint_latest.pth"]
for ckpt in checkpoint_files:
    ckpt_path = os.path.join(fold_dir, ckpt)
    if os.path.exists(ckpt_path):
        checkpoint_path = ckpt_path
        print(f"Found checkpoint: {ckpt}")
        break

# ============================================
# Evaluation Phase
# ============================================
print(f"\n{'='*60}")
print("EVALUATING ON TASK 2 VALIDATION SET")
print(f"{'='*60}")

pred_output_dir = os.path.join(base_dir, "evaluation_results_combined", "predictions")
os.makedirs(pred_output_dir, exist_ok=True)

# Generate predictions
print("Generating predictions for validation set...")
print(f"Validation cases: {', '.join(all_validation_cases)}")

success = generate_predictions(
    dataset_name=CONFIG['dataset_name'],
    plans_name=CONFIG['plans_name'],
    trainer_name=CONFIG['trainer_name'],
    output_dir=pred_output_dir,
    val_cases=all_validation_cases
)

if success:
    print("\nConverting 3-class predictions to binary and evaluating...")
    
    # Evaluate predictions (with 3-class to binary conversion)
    eval_results = evaluate_predictions_with_conversion(
        pred_dir=pred_output_dir,
        gt_dir=os.path.join(CONFIG['base_dir'], "nnUNet_raw", CONFIG['dataset_name'], "labelsTr"),
        val_cases=all_validation_cases
    )
    
    if eval_results:
        # Save evaluation results
        eval_save_path = os.path.join(base_dir, "evaluation_results_combined", "metrics.json")
        with open(eval_save_path, "w") as f:
            json.dump(eval_results, f, indent=4)
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS:")
        print(f"{'='*60}")
        print(f"  Mean Volumetric Dice: {eval_results['aggregates']['mean_volumetric_dice']:.4f}")
        print(f"  Mean Surface Dice (5mm): {eval_results['aggregates']['mean_surface_dice']:.4f}")
        print(f"  Mean Hausdorff95: {eval_results['aggregates']['mean_hausdorff95']:.2f} mm")
        print(f"  Mean MASD: {eval_results['aggregates']['mean_masd']:.2f} mm")
        print(f"  Tumor Burden RMSE: {eval_results['aggregates']['tumor_burden_rmse']:.2f} mm³")
        
        # Save comprehensive summary
        summary = {
            "approach": "Combined Task 1 and Task 2 training from scratch with domain adaptation",
            "architecture": f"ResEnc{CONFIG['resenc_variant']}",
            "domain_adaptation": {
                "enabled": CONFIG["use_domain_adaptation"],
                "method": CONFIG["adaptation_method"],
                "target_statistics_source": f"All {len(task2_images)} Task 2 images"
            },
            "training_composition": {
                "task1_labeled": task1_labeled_count,
                "task1_pseudo": task1_pseudo_count,
                "task1_background_only_excluded": background_only_count,
                "task2_training": task2_train_count,
                "total_training": len(all_training_cases)
            },
            "validation_composition": {
                "task2_validation": task2_val_count,
                "validation_cases": all_validation_cases
            },
            "hyperparameters": {
                "num_epochs": CONFIG['num_epochs'],
                "learning_rate": CONFIG['initial_lr']
            },
            "model_checkpoint": checkpoint_path,
            "evaluation_metrics": eval_results['aggregates'],
            "per_case_metrics": eval_results.get('per_case', {})
        }
        
        summary_path = os.path.join(base_dir, "evaluation_results_combined", "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nTraining summary saved to: {summary_path}")
    else:
        print("ERROR: Evaluation failed")
else:
    print("ERROR: Prediction generation failed")

# ============================================
# Final Summary
# ============================================
print("\n" + "="*80)
print("COMBINED TRAINING PIPELINE WITH DOMAIN ADAPTATION COMPLETED")
print("="*80)
print(f"\nFinal Summary:")
print(f"  Total training samples: {len(all_training_cases)}")
print(f"    - Task 1 labeled: {task1_labeled_count}")
print(f"    - Task 1 pseudo-labeled: {task1_pseudo_count}")
print(f"    - Task 2 training: {task2_train_count}")
print(f"  Validation samples: {len(all_validation_cases)} (Task 2 only)")
print(f"  Number of epochs: {CONFIG['num_epochs']}")
print(f"  Learning rate: {CONFIG['initial_lr']}")
print(f"  Domain Adaptation: {CONFIG['use_domain_adaptation']}")
if CONFIG["use_domain_adaptation"]:
    print(f"  Adaptation Method: {CONFIG['adaptation_method']}")

print(f"\nResults saved in:")
print(f"  - Model: {fold_dir}")
print(f"  - Evaluation: {os.path.join(base_dir, 'evaluation_results_combined')}")

print("\n" + "="*80)
print("IMPORTANT NOTES:")
print("="*80)
print("1. This script trains ResEncM from scratch on combined data")
print("2. Task 1 images are transformed to match Task 2 intensity distribution")
print(f"3. Domain adaptation method used: {CONFIG['adaptation_method']}")
print("4. Validation is performed only on 4 Task 2 samples:")
for i, case in enumerate(all_validation_cases, 1):
    print(f"   {i}. {case}")
print("5. To adjust epochs, learning rate, or adaptation method, modify CONFIG")
print("6. Ensure all data paths are correctly set up before running")

print("\nTraining pipeline with domain adaptation completed successfully!")
