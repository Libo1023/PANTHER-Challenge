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

# Import the FIXED evaluation function
sys.path.append('./')
from evaluate_local_fixed import evaluate_segmentation_performance

# ============================================
# PARSE COMMAND LINE ARGUMENTS
# ============================================
parser = argparse.ArgumentParser(description='Noisy Student Training Pipeline - Single Fold')
parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4],
                    help='Which fold to train (0-4)')
args = parser.parse_args()

SELECTED_FOLD = args.fold

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
    "num_epochs_student": 1600,
    "initial_lr": 0.003,
    
    # Training configuration - SINGLE FOLD
    "fold_to_train": SELECTED_FOLD,
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

print(f"Noisy Student Training Pipeline - FOLD {SELECTED_FOLD}")
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
os.makedirs(os.path.join(base_dir, f"evaluation_results_student_fold{SELECTED_FOLD}"), exist_ok=True)
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
# Define All 5-Fold Splits
# ============================================
ORIGINAL_SPLITS = [
    {  # Fold 0
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
    {  # Fold 1
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
    {  # Fold 2
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
    {  # Fold 3
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
    {  # Fold 4
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
    print("CREATING DATASET STRUCTURE (First time setup)")
    print("="*60)
    
    # ============================================
    # Create Custom Student Trainer
    # ============================================
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
    # Create Custom Splits
    # ============================================
    print("\nCreating custom 5-fold splits...")
    
    # Modify splits to add ALL pseudo cases to training
    modified_splits = []
    for fold_idx, split in enumerate(ORIGINAL_SPLITS):
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
    os.makedirs(preprocessed_dataset_dir, exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(modified_splits, f, indent=2)
    
    print(f"Saved modified splits to: {splits_path}")
    
    # Save dataset info for future runs
    dataset_info = {
        "labeled_cases": labeled_cases,
        "pseudo_cases": pseudo_cases,
        "background_only_count": background_only_count
    }
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=4)

else:
    print("\n" + "="*60)
    print("DATASET ALREADY EXISTS - SKIPPING CREATION")
    print("="*60)
    
    # Load existing dataset info
    info_path = os.path.join(dataset_dir, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            dataset_info = json.load(f)
            labeled_cases = dataset_info.get("labeled_cases", [])
            pseudo_cases = dataset_info.get("pseudo_cases", [])
            background_only_count = dataset_info.get("background_only_count", 0)
    
    # Load existing splits
    with open(splits_path, "r") as f:
        modified_splits = json.load(f)

# ============================================
# Student Training Phase for Selected Fold
# ============================================
print("\n" + "="*80)
print(f"NOISY STUDENT TRAINING - FOLD {SELECTED_FOLD}")
print("="*80)
print(f"Configuration:")
print(f"  Architecture: ResEnc{CONFIG['resenc_variant']}")
print(f"  Epochs: {CONFIG['num_epochs_student']}")
print(f"  Learning Rate: {CONFIG['initial_lr']}")
print(f"  Training samples: {len(modified_splits[SELECTED_FOLD]['train'])} "
      f"({len(ORIGINAL_SPLITS[SELECTED_FOLD]['train'])} labeled + {len(pseudo_cases)} pseudo)")
print(f"  Validation samples: {len(modified_splits[SELECTED_FOLD]['val'])}")
print("="*80)

# Train the selected fold
print(f"\n{'='*60}")
print(f"TRAINING FOLD {SELECTED_FOLD}")
print(f"{'='*60}")

train_cmd = [
    "nnUNetv2_train", 
    str(CONFIG['dataset_id']), 
    "3d_fullres", 
    str(SELECTED_FOLD), 
    "-p", CONFIG['plans_name'], 
    "-tr", CONFIG['trainer_name']
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

val_cases = modified_splits[SELECTED_FOLD]['val']
pred_output_dir = os.path.join(base_dir, f"evaluation_results_student_fold{SELECTED_FOLD}", "predictions")
os.makedirs(pred_output_dir, exist_ok=True)

# Generate predictions
print("Generating predictions for validation set...")
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
        eval_save_path = os.path.join(base_dir, f"evaluation_results_student_fold{SELECTED_FOLD}", "metrics.json")
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
            "approach": "Noisy student training without fine-tuning",
            "labeled_samples": len(ORIGINAL_SPLITS[SELECTED_FOLD]['train']),
            "pseudo_samples": len(pseudo_cases),
            "validation_samples": len(val_cases),
            "num_epochs": CONFIG['num_epochs_student'],
            "learning_rate": CONFIG['initial_lr'],
            "model_checkpoint": checkpoint_path,
            "evaluation_metrics": eval_results['aggregates']
        }
        
        summary_path = os.path.join(base_dir, f"evaluation_results_student_fold{SELECTED_FOLD}", "fold_summary.json")
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
print(f"FOLD {SELECTED_FOLD} TRAINING COMPLETED")
print("="*80)
print(f"\nSummary:")
print(f"  Training samples: {len(modified_splits[SELECTED_FOLD]['train'])}")
print(f"    - Labeled: {len(ORIGINAL_SPLITS[SELECTED_FOLD]['train'])}")
print(f"    - Pseudo-labeled: {len(pseudo_cases)}")
print(f"  Validation samples: {len(modified_splits[SELECTED_FOLD]['val'])}")
print(f"  Number of epochs: {CONFIG['num_epochs_student']}")
print(f"  Learning rate: {CONFIG['initial_lr']}")

print(f"\nResults saved in:")
print(f"  - Model: {fold_dir}")
print(f"  - Evaluation: {os.path.join(base_dir, f'evaluation_results_student_fold{SELECTED_FOLD}')}")

print(f"\nTo train other folds, run:")
for f in range(5):
    if f != SELECTED_FOLD:
        print(f"  python {sys.argv[0]} --fold {f}")

print("\nTraining pipeline completed successfully!")