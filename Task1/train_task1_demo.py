# ============================================
# CENTRALIZED CONFIGURATION - MODIFY ALL PARAMETERS HERE
# ============================================
CONFIG = {
    # Dataset configuration
    "dataset_id": 90,
    "dataset_name": "Dataset090_PantherTask1",

    # RESIDUAL ENCODER VARIANT SELECTION - Choose one: "M", "L", or "XL"
    "resenc_variant": "M",  # Options: "M" (9-11GB), "L" (24GB), "XL" (40GB)

    # Training hyperparameters
    "num_epochs": 300,  # Training will stop at this epoch
    "initial_lr": 0.005,  # Initial learning rate
    "num_iterations_per_epoch": 200,  # Iterations per epoch
    "num_val_iterations_per_epoch": 40,  # Validation iterations per epoch

    # Loss function weights
    "weight_ce": 1.0,  # Cross-entropy weight
    "weight_dice": 1.5,  # Dice weight

    # Data augmentation
    "rotation_degrees": 20,  # Max rotation in degrees for data augmentation

    # Evaluation settings
    "surface_tolerance_mm": 5,  # Tolerance for surface dice metric
    "hausdorff_percentile": 95,  # Percentile for Hausdorff distance

    # Training configuration
    "folds_to_train": [0, 1, 2, 3, 4],  # 5-fold cross-validation
    "trainer_name": "nnUNetTrainer_PANTHER_3Class_Optimized",

    # Paths (will be set based on base_dir)
    "base_dir": "/content/drive/MyDrive/PANTHER_nnUNet"
}

# Set ResEnc-specific configurations based on variant selection
RESENC_CONFIGS = {
    "M": {
        "planner_name": "nnUNetPlannerResEncM",
        "plans_name": "nnUNetResEncUNetMPlans",
        "expected_vram": "~9 GB",
        "expected_time": "~12 hours"
    },
    "L": {
        "planner_name": "nnUNetPlannerResEncL",
        "plans_name": "nnUNetResEncUNetLPlans",
        "expected_vram": "~23 GB",
        "expected_time": "~35 hours"
    },
    "XL": {
        "planner_name": "nnUNetPlannerResEncXL",
        "plans_name": "nnUNetResEncUNetXLPlans",
        "expected_vram": "~37 GB",
        "expected_time": "~66 hours"
    }
}

# Validate and set ResEnc configuration
if CONFIG["resenc_variant"] not in RESENC_CONFIGS:
    raise ValueError(f"Invalid ResEnc variant: {CONFIG['resenc_variant']}. Choose from: M, L, or XL")

# Add ResEnc-specific settings to CONFIG
CONFIG.update({
    "planner_name": RESENC_CONFIGS[CONFIG["resenc_variant"]]["planner_name"],
    "plans_name": RESENC_CONFIGS[CONFIG["resenc_variant"]]["plans_name"],
    "expected_vram": RESENC_CONFIGS[CONFIG["resenc_variant"]]["expected_vram"],
    "expected_time": RESENC_CONFIGS[CONFIG["resenc_variant"]]["expected_time"]
})

print(f"Selected ResEnc variant: {CONFIG['resenc_variant']}")
print(f"Expected VRAM usage: {CONFIG['expected_vram']}")
print(f"Expected training time per fold: {CONFIG['expected_time']} on A100")

# ============================================
# CELL 1: Mount Drive and Install Dependencies
# ============================================
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Change to working directory
import os
os.chdir('/content/drive/MyDrive/PANTHER_nnUNet')

# Install all required packages
!pip install nnunetv2 -q
!pip install SimpleITK -q
!pip install surface-distance -q
!pip install scipy -q
!pip install graphviz -q

print("All packages installed successfully!")

# ============================================
# CELL 2: Set Environment Variables and Create Directories
# ============================================
import os
import json
import shutil
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Base directory from config
base_dir = CONFIG["base_dir"]

# Set nnU-Net environment variables
os.environ['nnUNet_raw'] = f"{base_dir}/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = f"{base_dir}/nnUNet_preprocessed"
os.environ['nnUNet_results'] = f"{base_dir}/nnUNet_results"

# Create necessary directories
!mkdir -p {base_dir}/nnUNet_raw
!mkdir -p {base_dir}/nnUNet_preprocessed
!mkdir -p {base_dir}/nnUNet_results
!mkdir -p {base_dir}/evaluation_results
!mkdir -p {base_dir}/custom_trainers

print("Environment variables set:")
print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
print(f"nnUNet_results: {os.environ['nnUNet_results']}")

# ============================================
# CELL 3: Create Custom Trainer Class
# ============================================

# Create custom trainer file with configuration from CONFIG
custom_trainer_code = f'''
import torch
import numpy as np
from typing import List
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
import os
import json
from batchgenerators.utilities.file_and_folder_operations import join

class {CONFIG["trainer_name"]}(nnUNetTrainer):
    """Custom trainer for PANTHER Task 1 with 3-class training"""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Set training parameters from centralized config
        self.num_epochs = {CONFIG["num_epochs"]}
        self.initial_lr = {CONFIG["initial_lr"]}
        self.num_iterations_per_epoch = {CONFIG["num_iterations_per_epoch"]}
        self.num_val_iterations_per_epoch = {CONFIG["num_val_iterations_per_epoch"]}

        # Tracking for surface metrics
        self.surface_metrics_log = []

    def _build_loss(self):
        """Custom loss for 3-class segmentation"""
        # Multi-class segmentation loss (background, tumor, pancreas)
        loss = DC_and_CE_loss(
            {{'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5,
             'do_bg': False,  # Ignore background in dice calculation
             'ddp': self.is_ddp}},
            {{}},
            weight_ce={CONFIG["weight_ce"]},
            weight_dice={CONFIG["weight_dice"]},
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """Custom augmentation for brain MRI segmentation"""
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \\
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # Less aggressive rotation for brain MRI
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        if dim == 3:
            rotation_degrees = {CONFIG["rotation_degrees"]}
            rotation_for_DA = (-rotation_degrees / 360 * 2. * np.pi, rotation_degrees / 360 * 2. * np.pi)

        self.print_to_log_file(f'Custom rotation for brain MRI: +/- {CONFIG["rotation_degrees"]} degrees')

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def on_epoch_end(self):
        """Enhanced epoch end with progress monitoring"""
        super().on_epoch_end()

        # Print detailed progress every 10 epochs
        if self.current_epoch % 10 == 0:
            self.print_to_log_file("="*60)
            self.print_to_log_file(f"Progress Report - Epoch {{self.current_epoch}}/{{self.num_epochs}}")
            self.print_to_log_file(f"Current LR: {{self.optimizer.param_groups[0]['lr']:.6f}}")
            self.print_to_log_file(f"Train Loss: {{self.logger.my_fantastic_logging['train_losses'][-1]:.4f}}")
            self.print_to_log_file(f"Val Loss: {{self.logger.my_fantastic_logging['val_losses'][-1]:.4f}}")
            # Note: ema_fg_dice will be averaged across tumor and pancreas classes
            self.print_to_log_file(f"Pseudo Dice: {{self.logger.my_fantastic_logging['ema_fg_dice'][-1]:.4f}}")
            self.print_to_log_file("="*60)

    def on_train_end(self):
        """Enhanced train end with final metrics information"""
        super().on_train_end()

        self.print_to_log_file("="*60)
        self.print_to_log_file("Training completed! (3-class segmentation)")
        self.print_to_log_file(f"Total epochs trained: {{self.current_epoch}}")
        self.print_to_log_file(f"Best EMA Dice: {{self._best_ema:.4f}}")

        # Check which checkpoint files exist
        checkpoint_files = []
        if os.path.exists(join(self.output_folder, "checkpoint_best.pth")):
            checkpoint_files.append("checkpoint_best.pth")
        if os.path.exists(join(self.output_folder, "checkpoint_final.pth")):
            checkpoint_files.append("checkpoint_final.pth")
        if os.path.exists(join(self.output_folder, "checkpoint_latest.pth")):
            checkpoint_files.append("checkpoint_latest.pth")

        self.print_to_log_file(f"Available checkpoints: {{checkpoint_files}}")
        self.print_to_log_file("="*60)
'''

# Save custom trainer
trainer_path = f"{base_dir}/custom_trainers/{CONFIG['trainer_name']}.py"
os.makedirs(os.path.dirname(trainer_path), exist_ok=True)
with open(trainer_path, 'w') as f:
    f.write(custom_trainer_code)

print("Custom trainer created successfully at:")
print(trainer_path)

# Make the custom trainer discoverable
!cp {trainer_path} /usr/local/lib/python*/dist-packages/nnunetv2/training/nnUNetTrainer/

# ============================================
# CELL 4: Create Dataset Structure and Copy Data
# ============================================

# Create dataset directory structure
dataset_dir = f"{base_dir}/nnUNet_raw/{CONFIG['dataset_name']}"
!mkdir -p {dataset_dir}/imagesTr
!mkdir -p {dataset_dir}/labelsTr

print(f"Created dataset directory: {dataset_dir}")

# Copy and rename images to nnU-Net format
source_images = f"{base_dir}/data/PANTHER_Task1/ImagesTr"
source_labels = f"{base_dir}/data/PANTHER_Task1/LabelsTr"

dest_images = f"{dataset_dir}/imagesTr"
dest_labels = f"{dataset_dir}/labelsTr"

# Check if source data exists
if not os.path.exists(source_images):
    print(f"ERROR: Source images not found at {source_images}")
    print("Please ensure your original data is at:")
    print(f"  {base_dir}/data/PANTHER_Task1/ImagesTr")
    print(f"  {base_dir}/data/PANTHER_Task1/LabelsTr")
else:
    # Copy and rename images
    print("Copying images...")
    image_count = 0
    for img_file in Path(source_images).glob("*.mha"):
        case_id = img_file.stem.replace("_0001_0000", "").replace("_0001", "")
        dest_path = Path(dest_images) / f"{case_id}_0000.mha"
        shutil.copy2(img_file, dest_path)
        image_count += 1
        if image_count % 10 == 0:
            print(f"Progress: Copied {image_count} images...")

    print(f"Total images copied: {image_count}")

# ============================================
# CELL 5: Copy Multi-class Labels (No Conversion)
# ============================================

if os.path.exists(source_labels):
    print("\nCopying multi-class labels (keeping 3 classes)...")
    label_files = list(Path(source_labels).glob("*.mha"))

    for i, label_file in enumerate(label_files):
        # Simply copy the label file with proper naming
        case_id = label_file.stem.replace("_0001", "")
        dest_path = Path(dest_labels) / f"{case_id}.mha"
        shutil.copy2(label_file, dest_path)

        if (i + 1) % 10 == 0:
            print(f"Progress: Copied {i + 1}/{len(label_files)} labels...")

    print(f"Total labels copied: {len(label_files)}")

    # Verify the labels
    print("\nVerifying multi-class labels...")
    for i, label_file in enumerate(list(Path(dest_labels).glob("*.mha"))[:3]):
        img = sitk.ReadImage(str(label_file))
        arr = sitk.GetArrayFromImage(img)
        unique_values = np.unique(arr)
        print(f"  {label_file.name}: unique values = {unique_values}")
        print(f"    Class distribution: Background={np.sum(arr==0)}, Tumor={np.sum(arr==1)}, Pancreas={np.sum(arr==2)}")

# ============================================
# CELL 6: Create dataset.json for 3-class segmentation (FIXED)
# ============================================

# Count actual number of training cases
actual_num_training = len(list(Path(dest_images).glob("*_0000.mha")))
print(f"\nActual number of training cases found: {actual_num_training}")

dataset_json = {
    "channel_names": {
        "0": "MRI"
    },
    "labels": {
        "background": 0,
        "tumor": 1,
        "pancreas": 2
    },
    "numTraining": actual_num_training,  # FIXED: Use actual count instead of hardcoded 91
    "file_ending": ".mha"
}

dataset_json_path = f"{dataset_dir}/dataset.json"
with open(dataset_json_path, "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"Created dataset.json at: {dataset_json_path}")
print("Content:")
print(json.dumps(dataset_json, indent=4))

# ============================================
# CELL 7: Create splits_final.json (5-fold)
# ============================================

splits_content = [
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

# Verify splits total matches actual cases
total_cases_in_splits = len(set(splits_content[0]['train'] + splits_content[0]['val']))
print(f"\nTotal unique cases in splits: {total_cases_in_splits}")
print(f"Matches actual training cases: {total_cases_in_splits == actual_num_training}")

splits_path = f"{base_dir}/splits_final.json"
with open(splits_path, "w") as f:
    json.dump(splits_content, f, indent=4)

print(f"\nCreated 5-fold splits_final.json")
print(f"Fold 0: {len(splits_content[0]['train'])} train, {len(splits_content[0]['val'])} val")
print(f"Fold 1: {len(splits_content[1]['train'])} train, {len(splits_content[1]['val'])} val")
print(f"Fold 2: {len(splits_content[2]['train'])} train, {len(splits_content[2]['val'])} val")
print(f"Fold 3: {len(splits_content[3]['train'])} train, {len(splits_content[3]['val'])} val")
print(f"Fold 4: {len(splits_content[4]['train'])} train, {len(splits_content[4]['val'])} val")

# ============================================
# CELL 8: Run nnU-Net Planning and Preprocessing
# ============================================

print("\n" + "="*60)
print(f"Running nnU-Net planning and preprocessing with ResEnc{CONFIG['resenc_variant']} for 3-class segmentation...")
print("This will take several minutes...")
print("="*60)

# Check if data has already been preprocessed
preprocessed_data_path = f"{base_dir}/nnUNet_preprocessed/{CONFIG['dataset_name']}"
if os.path.exists(preprocessed_data_path) and os.path.exists(f"{preprocessed_data_path}/gt_segmentations"):
    print("\nPreprocessed data already exists.")
    print(f"Running only planning with ResEnc{CONFIG['resenc_variant']} planner to avoid re-preprocessing...")
    !nnUNetv2_plan_experiment -d {CONFIG["dataset_id"]} -pl {CONFIG["planner_name"]}
else:
    print(f"\nRunning full planning and preprocessing with ResEnc{CONFIG['resenc_variant']}...")
    !nnUNetv2_plan_and_preprocess -d {CONFIG["dataset_id"]} -pl {CONFIG["planner_name"]} --verify_dataset_integrity

# Copy splits to preprocessed directory
preprocessed_splits = f"{base_dir}/nnUNet_preprocessed/{CONFIG['dataset_name']}/splits_final.json"
!cp {splits_path} {preprocessed_splits}
print(f"\nCopied splits to preprocessed directory")

# ============================================
# CELL 9: Verify ResEnc Architecture Settings
# ============================================

# Verify the ResEnc plans were created for 3-class segmentation
plans_path = f"{base_dir}/nnUNet_preprocessed/{CONFIG['dataset_name']}/{CONFIG['plans_name']}.json"

if os.path.exists(plans_path):
    with open(plans_path, 'r') as f:
        plans = json.load(f)

    print(f"\nResEnc{CONFIG['resenc_variant']} 3d_fullres architecture (3-class):")
    print(f"Features per stage: {plans['configurations']['3d_fullres']['architecture']['arch_kwargs']['features_per_stage']}")
    print(f"Batch size: {plans['configurations']['3d_fullres']['batch_size']}")
    print(f"Network class: {plans['configurations']['3d_fullres']['architecture']['network_class_name']}")
    print(f"Number of output channels: 3 (background, tumor, pancreas)")
    print(f"\nNOTE: ResEnc{CONFIG['resenc_variant']} uses its optimized settings - no manual modification needed!")
else:
    print(f"\nWARNING: ResEnc{CONFIG['resenc_variant']} plans not found at {plans_path}")
    print("The planning step may have failed. Please check the output above.")

# ============================================
# CELL 10: Define Enhanced Evaluation Functions (Binary Evaluation from 3-class Predictions)
# ============================================

from surface_distance import metrics as surface_metrics

def find_best_checkpoint(fold_dir):
    """Find the best available checkpoint in fold directory"""
    print(f"  Looking for checkpoints in: {fold_dir}")

    # First, list all files in the directory to debug
    if os.path.exists(fold_dir):
        all_files = os.listdir(fold_dir)
        checkpoint_files = [f for f in all_files if f.endswith('.pth')]
        print(f"  Available checkpoint files: {checkpoint_files}")
    else:
        print(f"  ERROR: Fold directory does not exist!")
        return None

    possible_checkpoints = ['checkpoint_best.pth', 'checkpoint_final.pth', 'checkpoint_latest.pth']

    for checkpoint in possible_checkpoints:
        checkpoint_path = os.path.join(fold_dir, checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"  Found checkpoint: {checkpoint} at {checkpoint_path}")
            # Return with .pth extension
            return checkpoint

    # Check for epoch checkpoints
    epoch_checkpoints = [f for f in checkpoint_files if f.startswith('checkpoint_epoch_')]
    if epoch_checkpoints:
        # Sort by epoch number and return the latest
        epoch_checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        print(f"  Found epoch checkpoint: {epoch_checkpoints[-1]}")
        return epoch_checkpoints[-1]

    print(f"  WARNING: No checkpoint files found in {fold_dir}")
    return None

def generate_predictions_with_best_checkpoint(fold, results_base, gt_dir):
    """Always generate predictions using the best checkpoint"""
    pred_dir = f"{results_base}/fold_{fold}/validation_best"

    # Define preprocessed splits path
    preprocessed_splits = f"{CONFIG['base_dir']}/nnUNet_preprocessed/{CONFIG['dataset_name']}/splits_final.json"

    # Always regenerate to ensure we use best checkpoint
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.makedirs(pred_dir, exist_ok=True)

    # Find the best checkpoint
    fold_dir = f"{results_base}/fold_{fold}"
    best_checkpoint = find_best_checkpoint(fold_dir)

    if not best_checkpoint:
        print(f"  ERROR: No checkpoint found for fold {fold}")
        return None

    print(f"  Using checkpoint: {best_checkpoint}")

    # Verify checkpoint file exists
    checkpoint_path = os.path.join(fold_dir, best_checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"  ERROR: Checkpoint file does not exist: {checkpoint_path}")
        return None
    else:
        print(f"  Checkpoint file verified at: {checkpoint_path}")

    # Get validation case IDs from splits
    with open(preprocessed_splits, 'r') as f:
        splits = json.load(f)
    val_cases = splits[fold]['val']

    print(f"  Generating predictions for {len(val_cases)} validation cases...")

    # Create temporary directory with raw images for prediction
    temp_val_dir = f"{CONFIG['base_dir']}/temp_val_fold_{fold}"
    if os.path.exists(temp_val_dir):
        shutil.rmtree(temp_val_dir)
    os.makedirs(temp_val_dir, exist_ok=True)

    # Copy validation images to temp directory
    copied_count = 0
    for case in val_cases:
        src_file = f"{CONFIG['base_dir']}/nnUNet_raw/{CONFIG['dataset_name']}/imagesTr/{case}_0000.mha"
        if os.path.exists(src_file):
            shutil.copy2(src_file, temp_val_dir)
            copied_count += 1
        else:
            print(f"  Warning: Could not find image for case {case}")

    print(f"  Copied {copied_count} validation images to temp directory")

    # Run prediction with ResEnc plans
    pred_cmd = f"nnUNetv2_predict -i {temp_val_dir} -o {pred_dir} -d {CONFIG['dataset_id']} -p {CONFIG['plans_name']} -c 3d_fullres -f {fold} -tr {CONFIG['trainer_name']}"

    print(f"  Running prediction command (with ResEnc{CONFIG['resenc_variant']} plans)...")
    print(f"  Command: {pred_cmd}")
    !{pred_cmd}

    # Clean up temp directory
    shutil.rmtree(temp_val_dir)

    # Verify predictions were generated
    if os.path.exists(pred_dir):
        pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz') or f.endswith('.mha')]
        print(f"  Generated {len(pred_files)} prediction files")
        if len(pred_files) == 0:
            print(f"  WARNING: No prediction files were generated!")
            return None
    else:
        print(f"  ERROR: Prediction directory was not created")
        return None

    return pred_dir

def extract_tumor_from_multiclass(multiclass_pred):
    """Extract tumor class (label 1) from multi-class prediction"""
    # The prediction has values 0, 1, 2 for background, tumor, pancreas
    # We want binary mask where 1 = tumor, 0 = everything else
    return (multiclass_pred == 1).astype(np.uint8)

def evaluate_fold_predictions(pred_dir, gt_dir, fold_num, output_dir):
    """Evaluate predictions and compute all 5 metrics (binary evaluation from 3-class predictions)"""
    print(f"\nEvaluating Fold {fold_num} (binary metrics from 3-class predictions)...")

    # Check if prediction directory exists
    if not os.path.exists(pred_dir):
        print(f"  Prediction directory not found: {pred_dir}")
        return None

    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.nii.gz') or f.endswith('.mha')]

    if not pred_files:
        print(f"  No prediction files found in {pred_dir}")
        return None

    print(f"  Found {len(pred_files)} prediction files")

    metrics_list = []
    processed_count = 0

    for pred_file in pred_files:
        # Extract case ID
        if pred_file.endswith('.nii.gz'):
            case_id = pred_file[:-7]
        else:
            case_id = pred_file[:-4]

        # Find ground truth
        gt_file = None
        for ext in ['.mha', '.nii.gz']:
            gt_path = os.path.join(gt_dir, case_id + ext)
            if os.path.exists(gt_path):
                gt_file = gt_path
                break

        if gt_file is None:
            continue

        pred_path = os.path.join(pred_dir, pred_file)

        try:
            # Load images
            pred_img = sitk.ReadImage(pred_path)
            gt_img = sitk.ReadImage(gt_file)

            mask_pred_multiclass = sitk.GetArrayFromImage(pred_img)
            mask_gt_multiclass = sitk.GetArrayFromImage(gt_img)
            spacing = gt_img.GetSpacing()

            # Extract tumor class only for evaluation
            mask_pred = extract_tumor_from_multiclass(mask_pred_multiclass)
            mask_gt = extract_tumor_from_multiclass(mask_gt_multiclass)

            # Convert to boolean for surface metrics
            mask_pred_bool = mask_pred.astype(bool)
            mask_gt_bool = mask_gt.astype(bool)

            # Compute metrics
            if np.all(mask_pred_bool == 0) or np.all(mask_pred_bool == 1):
                max_dist = np.linalg.norm(np.array(mask_gt_bool.shape) * np.array(spacing))
                metrics = {
                    "case_id": case_id,
                    "volumetric_dice": 0.0,
                    "surface_dice_5mm": 0.0,
                    "hausdorff95": max_dist,
                    "masd": max_dist,
                    "gt_volume": np.sum(mask_gt_bool) * np.prod(spacing),
                    "pred_volume": np.sum(mask_pred_bool) * np.prod(spacing)
                }
            else:
                surface_distances = surface_metrics.compute_surface_distances(
                    mask_gt_bool, mask_pred_bool, spacing_mm=spacing
                )

                dice = surface_metrics.compute_dice_coefficient(mask_gt_bool, mask_pred_bool)
                surf_dice = surface_metrics.compute_surface_dice_at_tolerance(
                    surface_distances, tolerance_mm=CONFIG["surface_tolerance_mm"]
                )
                hausdorff95 = surface_metrics.compute_robust_hausdorff(
                    surface_distances, percent=CONFIG["hausdorff_percentile"]
                )
                avg_gt_to_pred, avg_pred_to_gt = surface_metrics.compute_average_surface_distance(
                    surface_distances
                )
                masd = (avg_gt_to_pred + avg_pred_to_gt) / 2.0

                voxel_volume = np.prod(spacing)
                gt_volume = np.sum(mask_gt_bool) * voxel_volume
                pred_volume = np.sum(mask_pred_bool) * voxel_volume

                metrics = {
                    "case_id": case_id,
                    "volumetric_dice": dice,
                    "surface_dice_5mm": surf_dice,
                    "hausdorff95": hausdorff95,
                    "masd": masd,
                    "gt_volume": gt_volume,
                    "pred_volume": pred_volume
                }

            metrics_list.append(metrics)
            processed_count += 1

            if processed_count % 5 == 0:
                print(f"    Progress: Evaluated {processed_count} cases...")

        except Exception as e:
            print(f"    Error processing {case_id}: {str(e)}")

    # Aggregate metrics
    if metrics_list:
        mean_dice = np.mean([m["volumetric_dice"] for m in metrics_list])
        mean_surf_dice = np.mean([m["surface_dice_5mm"] for m in metrics_list])
        mean_hausdorff95 = np.mean([m["hausdorff95"] for m in metrics_list])
        mean_masd = np.mean([m["masd"] for m in metrics_list])

        gt_vols = np.array([m["gt_volume"] for m in metrics_list])
        pred_vols = np.array([m["pred_volume"] for m in metrics_list])
        rmse_volume = np.sqrt(np.mean((pred_vols - gt_vols) ** 2))

        # Calculate standard deviations
        std_dice = np.std([m["volumetric_dice"] for m in metrics_list])
        std_surf_dice = np.std([m["surface_dice_5mm"] for m in metrics_list])
        std_hausdorff95 = np.std([m["hausdorff95"] for m in metrics_list])
        std_masd = np.std([m["masd"] for m in metrics_list])

        results = {
            "fold": fold_num,
            "num_cases": len(metrics_list),
            "config": {
                "architecture": f"ResEnc{CONFIG['resenc_variant']}",
                "training_mode": "3-class segmentation",
                "evaluation_mode": "binary (tumor only)",
                "epochs": CONFIG["num_epochs"],
                "learning_rate": CONFIG["initial_lr"],
                "planner": CONFIG["planner_name"],
                "plans": CONFIG["plans_name"]
            },
            "aggregate_metrics": {
                "mean_volumetric_dice": mean_dice,
                "std_volumetric_dice": std_dice,
                "mean_surface_dice_5mm": mean_surf_dice,
                "std_surface_dice_5mm": std_surf_dice,
                "mean_hausdorff95": mean_hausdorff95,
                "std_hausdorff95": std_hausdorff95,
                "mean_masd": mean_masd,
                "std_masd": std_masd,
                "tumor_burden_rmse": rmse_volume
            },
            "case_metrics": metrics_list
        }

        # Save detailed results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"fold_{fold_num}_evaluation_3class_resenc{CONFIG['resenc_variant'].lower()}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n  FOLD {fold_num} RESULTS (ResEnc{CONFIG['resenc_variant']} - 3-class training, binary eval):")
        print(f"  Cases evaluated: {len(metrics_list)}")
        print(f"  1. Volumetric Dice: {mean_dice:.4f} +/- {std_dice:.4f}")
        print(f"  2. Surface Dice 5mm: {mean_surf_dice:.4f} +/- {std_surf_dice:.4f}")
        print(f"  3. Hausdorff95: {mean_hausdorff95:.2f} +/- {std_hausdorff95:.2f} mm")
        print(f"  4. MASD: {mean_masd:.2f} +/- {std_masd:.2f} mm")
        print(f"  5. Volume RMSE: {rmse_volume:.2f} mm3")

        return results

    return None

# ============================================
# CELL 11: Train and Evaluate Each Fold
# ============================================

print("\n" + "="*80)
print(f"STARTING TRAINING WITH RESIDUAL ENCODER UNET {CONFIG['resenc_variant']} (ResEnc{CONFIG['resenc_variant']})")
print("3-CLASS SEGMENTATION TRAINING WITH BINARY EVALUATION")
print("="*80)
print(f"Configuration Summary:")
print(f"  Architecture: ResEnc{CONFIG['resenc_variant']} (Residual Encoder UNet)")
print(f"  Training mode: 3-class segmentation (background, tumor, pancreas)")
print(f"  Evaluation mode: Binary (tumor only)")
print(f"  Cross-validation: 5-fold")
print(f"  Planner: {CONFIG['planner_name']}")
print(f"  Plans: {CONFIG['plans_name']}")
print(f"  Epochs: {CONFIG['num_epochs']}")
print(f"  Initial LR: {CONFIG['initial_lr']}")
print(f"  Iterations per epoch: {CONFIG['num_iterations_per_epoch']}")
print(f"  Validation iterations: {CONFIG['num_val_iterations_per_epoch']}")
print(f"  Loss weights - CE: {CONFIG['weight_ce']}, Dice: {CONFIG['weight_dice']}")
print(f"  Rotation augmentation: +/- {CONFIG['rotation_degrees']} degrees")
print(f"  Trainer: {CONFIG['trainer_name']}")
print(f"  Expected VRAM usage: {CONFIG['expected_vram']}")
print(f"  Expected training time per fold: {CONFIG['expected_time']} on A100")
print("="*80)

# Train each fold
all_results = []

for fold in CONFIG["folds_to_train"]:
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold} with ResEnc{CONFIG['resenc_variant']} (3-class)")
    print(f"{'='*60}")

    # Training command with ResEnc plans
    train_cmd = f"nnUNetv2_train {CONFIG['dataset_id']} 3d_fullres {fold} -p {CONFIG['plans_name']} -tr {CONFIG['trainer_name']}"
    print(f"\nCommand: {train_cmd}")
    print(f"Training will run for {CONFIG['num_epochs']} epochs")
    print("Progress will be monitored every 10 epochs\n")

    # Run training
    !{train_cmd}

    # Find results directory
    print(f"\nPreparing evaluation for fold {fold}...")

    possible_results_bases = [
        f"{base_dir}/nnUNet_results/{CONFIG['dataset_name']}/{CONFIG['trainer_name']}__{CONFIG['plans_name']}__3d_fullres",
        f"{base_dir}/nnUNet_results/{CONFIG['dataset_name']}/nnUNetTrainer__{CONFIG['plans_name']}__3d_fullres"
    ]

    results_base = None
    for base in possible_results_bases:
        if os.path.exists(base):
            results_base = base
            print(f"  Found results directory: {results_base}")
            break

    if results_base is None:
        print(f"  ERROR: Could not find results directory for fold {fold}")
        continue

    # Generate predictions with best checkpoint
    gt_dir = f"{base_dir}/nnUNet_raw/{CONFIG['dataset_name']}/labelsTr"
    pred_dir = generate_predictions_with_best_checkpoint(fold, results_base, gt_dir)

    if pred_dir is None:
        print(f"  ERROR: Could not generate predictions for fold {fold}")
        continue

    # Evaluate predictions (extracting tumor class only)
    eval_dir = f"{base_dir}/evaluation_results"
    fold_results = evaluate_fold_predictions(pred_dir, gt_dir, fold, eval_dir)

    if fold_results:
        all_results.append(fold_results)

# ============================================
# CELL 12: Final Summary and Analysis
# ============================================

print("\n" + "="*80)
print(f"FINAL SUMMARY - ResEnc{CONFIG['resenc_variant']} RESULTS ACROSS ALL FOLDS")
print("3-CLASS TRAINING WITH BINARY EVALUATION")
print("="*80)

if all_results:
    # Collect metrics across folds
    metric_names = [
        "mean_volumetric_dice",
        "mean_surface_dice_5mm",
        "mean_hausdorff95",
        "mean_masd",
        "tumor_burden_rmse"
    ]

    print(f"\nPER-FOLD RESULTS (ResEnc{CONFIG['resenc_variant']} - 3-class training):")
    for result in all_results:
        fold = result['fold']
        print(f"\nFold {fold} ({result['num_cases']} cases):")
        for metric_name in metric_names:
            value = result['aggregate_metrics'][metric_name]
            if 'std_' + metric_name.replace('mean_', '') in result['aggregate_metrics']:
                std = result['aggregate_metrics']['std_' + metric_name.replace('mean_', '')]
                print(f"  {metric_name}: {value:.4f} +/- {std:.4f}")
            else:
                print(f"  {metric_name}: {value:.4f}")

    print("\nMEAN +/- STD ACROSS FOLDS:")
    for metric_name in metric_names:
        values = [r['aggregate_metrics'][metric_name] for r in all_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric_name}: {mean_val:.4f} +/- {std_val:.4f}")

    # Save summary
    summary = {
        "architecture": f"ResEnc{CONFIG['resenc_variant']}",
        "training_mode": "3-class segmentation",
        "evaluation_mode": "binary (tumor only)",
        "configuration": CONFIG,
        "num_folds": len(all_results),
        "cross_fold_metrics": {}
    }

    for metric_name in metric_names:
        values = [r['aggregate_metrics'][metric_name] for r in all_results]
        summary["cross_fold_metrics"][metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values
        }

    summary_path = f"{base_dir}/evaluation_results/final_summary_3class_resenc{CONFIG['resenc_variant'].lower()}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")

    # Print configuration used
    print("\nConfiguration used:")
    print(f"  Architecture: ResEnc{CONFIG['resenc_variant']} (Residual Encoder UNet)")
    print(f"  Training: 3-class segmentation (background, tumor, pancreas)")
    print(f"  Evaluation: Binary (tumor class only)")
    print(f"  Cross-validation: 5-fold")
    print(f"  Epochs: {CONFIG['num_epochs']}")
    print(f"  Initial LR: {CONFIG['initial_lr']}")
    print(f"  Iterations: {CONFIG['num_iterations_per_epoch']} train / {CONFIG['num_val_iterations_per_epoch']} val")
    print(f"  CE/Dice weights: {CONFIG['weight_ce']}/{CONFIG['weight_dice']}")
    print(f"  Rotation augmentation: +/- {CONFIG['rotation_degrees']} degrees")

print("\nPipeline completed successfully!")
print(f"All results saved in: {base_dir}/evaluation_results/")
print(f"\nTraining logs for each fold can be found in: {results_base}/fold_X/training_log.txt")
