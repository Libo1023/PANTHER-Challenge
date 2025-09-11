#!/usr/bin/env python3
"""
nnU-Net 5-Fold Ensemble Inference Script with Detailed Progress Monitoring
Generates 3-class segmentation predictions for labeled and unlabeled images
"""

import os
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import subprocess
import shutil
import argparse
from datetime import datetime
import logging
import time
import sys

# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    # Dataset configuration
    "dataset_id": 90,
    "dataset_name": "Dataset090_PantherTask1",
    
    # Model configuration (from training)
    "resenc_variant": "M",
    "planner_name": "nnUNetPlannerResEncM",
    "plans_name": "nnUNetResEncUNetMPlans",
    "trainer_name": "nnUNetTrainer_PANTHER_3Class_Optimized",
    
    # Inference configuration
    "folds": [0, 1, 2, 3, 4],  # All 5 folds for ensemble
    "ensemble_method": "softmax_averaging",  # Method for combining predictions
    
    # Base paths
    "base_dir": ".",
    
    # Input paths
    "labeled_images_dir": "./data/PANTHER_Task1/ImagesTr",
    "unlabeled_images_dir": "./data/PANTHER_Task1/ImagesTr_unlabeled",
    
    # Output paths
    "output_labeled_dir": "./data/PANTHER_Task1/PredictionsTr_3class",
    "output_unlabeled_dir": "./data/PANTHER_Task1/PredictionsTr_unlabeled_3class",
    "log_dir": "./data/PANTHER_Task1/inference_logs",
    
    # Processing options
    "save_softmax": False,  # Whether to save softmax probabilities
    "verbose": True
}

# ============================================
# Progress Monitoring Class
# ============================================
class ProgressMonitor:
    """Class to handle detailed progress monitoring"""
    
    def __init__(self, total_images, set_name="images"):
        self.total_images = total_images
        self.set_name = set_name
        self.start_time = time.time()
        self.current_image = 0
        
    def update(self, image_name, status="Processing"):
        """Update progress for current image"""
        self.current_image += 1
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.current_image if self.current_image > 0 else 0
        eta = avg_time * (self.total_images - self.current_image)
        
        print(f"\n[{self.current_image}/{self.total_images}] {status}: {image_name}")
        print(f"  Progress: {self.current_image/self.total_images*100:.1f}%")
        print(f"  Elapsed: {self.format_time(elapsed)} | ETA: {self.format_time(eta)}")
        
    def format_time(self, seconds):
        """Format seconds to readable time"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"

# ============================================
# Setup Logging
# ============================================
def setup_logging(log_file):
    """Setup logging configuration"""
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ============================================
# Utility Functions
# ============================================
def set_environment_variables():
    """Set required nnU-Net environment variables"""
    os.environ['nnUNet_raw'] = os.path.join(CONFIG["base_dir"], "nnUNet_raw")
    os.environ['nnUNet_preprocessed'] = os.path.join(CONFIG["base_dir"], "nnUNet_preprocessed")
    os.environ['nnUNet_results'] = os.path.join(CONFIG["base_dir"], "nnUNet_results")
    
    # Suppress nnU-Net verbose output during inference
    os.environ['nnUNet_verbose'] = "0"
    
    return {
        'nnUNet_raw': os.environ['nnUNet_raw'],
        'nnUNet_preprocessed': os.environ['nnUNet_preprocessed'],
        'nnUNet_results': os.environ['nnUNet_results']
    }

def find_model_directory():
    """Find the trained model directory"""
    possible_paths = [
        os.path.join(CONFIG["base_dir"], "nnUNet_results", CONFIG["dataset_name"], 
                     f"{CONFIG['trainer_name']}__{CONFIG['plans_name']}__3d_fullres"),
        os.path.join(CONFIG["base_dir"], "nnUNet_results", CONFIG["dataset_name"], 
                     f"nnUNetTrainer__{CONFIG['plans_name']}__3d_fullres")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Could not find model directory. Looked in: {possible_paths}")

def verify_checkpoints(model_dir, logger):
    """Verify all fold checkpoints exist"""
    missing_folds = []
    checkpoint_info = {}
    
    logger.info("\nVerifying model checkpoints:")
    
    for fold in CONFIG["folds"]:
        fold_dir = os.path.join(model_dir, f"fold_{fold}")
        
        if not os.path.exists(fold_dir):
            missing_folds.append(fold)
            logger.error(f"  Fold {fold}: Directory not found!")
            continue
        
        # Find best checkpoint
        checkpoint_priority = ['checkpoint_best.pth', 'checkpoint_final.pth', 'checkpoint_latest.pth']
        checkpoint_found = None
        
        for checkpoint in checkpoint_priority:
            checkpoint_path = os.path.join(fold_dir, checkpoint)
            if os.path.exists(checkpoint_path):
                checkpoint_found = checkpoint
                break
        
        if checkpoint_found:
            checkpoint_info[fold] = checkpoint_found
            logger.info(f"  Fold {fold}: ✓ Found {checkpoint_found}")
        else:
            missing_folds.append(fold)
            logger.error(f"  Fold {fold}: ✗ No checkpoint found!")
    
    if missing_folds:
        raise FileNotFoundError(f"Missing checkpoints for folds: {missing_folds}")
    
    logger.info(f"\nAll {len(checkpoint_info)} fold checkpoints verified successfully!")
    return checkpoint_info

def get_image_list(image_dir, pattern="*.mha"):
    """Get list of images from directory"""
    image_files = list(Path(image_dir).glob(pattern))
    return sorted(image_files)

def extract_case_id(filename):
    """Extract case ID from filename"""
    stem = Path(filename).stem
    case_id = stem.replace("_0000", "")
    return case_id

# ============================================
# Inference Functions
# ============================================
def run_single_image_inference(image_path, fold, output_dir, model_dir, logger):
    """Run inference for a single image on a single fold"""
    case_id = extract_case_id(image_path.name)
    
    # Create temporary directories
    temp_input = os.path.join(CONFIG["base_dir"], f"temp_input_fold{fold}")
    temp_output = os.path.join(CONFIG["base_dir"], f"temp_output_fold{fold}")
    
    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)
    
    try:
        # Copy single image to temp input
        shutil.copy2(image_path, temp_input)
        
        # Build inference command
        cmd = [
            "nnUNetv2_predict",
            "-i", temp_input,
            "-o", temp_output,
            "-d", str(CONFIG["dataset_id"]),
            "-p", CONFIG["plans_name"],
            "-c", "3d_fullres",
            "-f", str(fold),
            "-tr", CONFIG["trainer_name"],
            "--save_probabilities",
            "--disable_progress_bar"  # Disable nnU-Net's progress bar
        ]
        
        # Run inference (suppress output)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Move results to fold output directory
        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Move prediction and probability files
        for file in os.listdir(temp_output):
            src = os.path.join(temp_output, file)
            dst = os.path.join(fold_output_dir, file)
            shutil.move(src, dst)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"    Fold {fold} failed: {e.stderr}")
        return False
        
    finally:
        # Clean up
        if os.path.exists(temp_input):
            shutil.rmtree(temp_input)
        if os.path.exists(temp_output):
            shutil.rmtree(temp_output)

def load_softmax_probabilities(case_id, fold_dir):
    """Load softmax probabilities for a case from fold predictions"""
    # Look for probability file (.npz)
    prob_file = os.path.join(fold_dir, f"{case_id}.npz")
    
    if os.path.exists(prob_file):
        data = np.load(prob_file)
        return data['probabilities']
    else:
        # If no probability file, load the prediction and convert to one-hot
        pred_file = os.path.join(fold_dir, f"{case_id}.nii.gz")
        if not os.path.exists(pred_file):
            pred_file = os.path.join(fold_dir, f"{case_id}.mha")
        
        if os.path.exists(pred_file):
            pred_img = sitk.ReadImage(pred_file)
            pred_array = sitk.GetArrayFromImage(pred_img)
            
            # Convert to one-hot encoding (3 classes)
            num_classes = 3
            one_hot = np.zeros((num_classes,) + pred_array.shape, dtype=np.float32)
            for c in range(num_classes):
                one_hot[c] = (pred_array == c).astype(np.float32)
            
            return one_hot
        else:
            raise FileNotFoundError(f"No prediction found for {case_id} in {fold_dir}")

def ensemble_predictions_for_image(case_id, fold_dirs, reference_image_path, output_path, logger):
    """Ensemble predictions from all folds for a single case"""
    
    # Load reference image for metadata
    ref_img = sitk.ReadImage(str(reference_image_path))
    
    # Collect softmax probabilities from all folds
    all_probs = []
    successful_folds = []
    
    for fold in sorted(fold_dirs.keys()):
        fold_dir = fold_dirs[fold]
        try:
            probs = load_softmax_probabilities(case_id, fold_dir)
            
            # Ensure correct shape (C, D, H, W)
            if probs.shape[0] != 3:
                probs = np.transpose(probs, (3, 0, 1, 2))
            
            all_probs.append(probs)
            successful_folds.append(fold)
            
        except Exception as e:
            logger.warning(f"    Could not load fold {fold} predictions: {e}")
    
    if not all_probs:
        raise ValueError(f"No predictions found for {case_id}")
    
    print(f"    Ensembling {len(successful_folds)} fold predictions: {successful_folds}")
    
    # Average softmax probabilities
    avg_probs = np.mean(all_probs, axis=0)
    
    # Get final prediction by argmax
    final_pred = np.argmax(avg_probs, axis=0).astype(np.uint8)
    
    # Create output image with correct metadata
    output_img = sitk.GetImageFromArray(final_pred)
    output_img.CopyInformation(ref_img)
    
    # Save prediction
    sitk.WriteImage(output_img, str(output_path))
    
    # Optionally save softmax probabilities
    if CONFIG["save_softmax"]:
        prob_path = output_path.with_suffix('.npz')
        np.savez_compressed(prob_path, probabilities=avg_probs)
    
    # Return class distribution for logging
    unique, counts = np.unique(final_pred, return_counts=True)
    class_dist = dict(zip(unique, counts))
    
    return class_dist, len(successful_folds)

def process_single_image(image_path, output_dir, model_dir, logger, progress_monitor):
    """Process a single image through all folds and ensemble"""
    case_id = extract_case_id(image_path.name)
    output_path = Path(output_dir) / f"{case_id}.mha"
    
    # Update progress
    progress_monitor.update(image_path.name, "Starting")
    
    # Skip if already processed
    if output_path.exists():
        print(f"    Skipping - already processed")
        return True, None
    
    # Run inference for each fold
    print(f"    Running 5-fold inference:")
    fold_dirs = {}
    successful_folds = 0
    
    for fold in CONFIG["folds"]:
        print(f"      Fold {fold}: ", end='', flush=True)
        start_time = time.time()
        
        success = run_single_image_inference(image_path, fold, output_dir, model_dir, logger)
        
        if success:
            print(f"✓ ({time.time()-start_time:.1f}s)")
            fold_dirs[fold] = os.path.join(output_dir, f"fold_{fold}")
            successful_folds += 1
        else:
            print(f"✗ Failed")
    
    if successful_folds == 0:
        logger.error(f"    All folds failed for {case_id}")
        return False, None
    
    # Ensemble predictions
    print(f"    Ensembling predictions...")
    try:
        class_dist, num_folds = ensemble_predictions_for_image(
            case_id, fold_dirs, image_path, output_path, logger
        )
        
        # Log results
        print(f"    Success! Used {num_folds} folds")
        print(f"    Class distribution: BG={class_dist.get(0,0):,} | Tumor={class_dist.get(1,0):,} | Pancreas={class_dist.get(2,0):,}")
        print(f"    Saved to: {output_path.name}")
        
        return True, class_dist
        
    except Exception as e:
        logger.error(f"    Ensemble failed: {e}")
        return False, None

def process_image_set(image_files, output_dir, model_dir, logger, set_name="images"):
    """Process a set of images with detailed progress monitoring"""
    logger.info(f"\nProcessing {len(image_files)} {set_name}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize progress monitor
    progress = ProgressMonitor(len(image_files), set_name)
    
    # Process statistics
    successful = 0
    failed = 0
    class_distributions = []
    
    # Process each image
    for image_path in image_files:
        success, class_dist = process_single_image(
            image_path, output_dir, model_dir, logger, progress
        )
        
        if success:
            successful += 1
            if class_dist:
                class_distributions.append(class_dist)
        else:
            failed += 1
    
    # Clean up fold directories
    print("\nCleaning up temporary fold directories...")
    for fold in CONFIG["folds"]:
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        if os.path.exists(fold_dir):
            shutil.rmtree(fold_dir)
    
    # Summary statistics
    logger.info(f"\n{set_name.upper()} PROCESSING COMPLETE:")
    logger.info(f"  Successful: {successful}/{len(image_files)}")
    if failed > 0:
        logger.info(f"  Failed: {failed}")
    
    # Average class distribution
    if class_distributions:
        avg_dist = {}
        for c in range(3):
            counts = [d.get(c, 0) for d in class_distributions]
            avg_dist[c] = np.mean(counts)
        
        logger.info("\n  Average voxel distribution per image:")
        logger.info(f"    Background: {avg_dist.get(0, 0):,.0f} voxels")
        logger.info(f"    Tumor: {avg_dist.get(1, 0):,.0f} voxels") 
        logger.info(f"    Pancreas: {avg_dist.get(2, 0):,.0f} voxels")
    
    return successful

# ============================================
# Main Execution
# ============================================
def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="nnU-Net 5-fold ensemble inference")
    parser.add_argument("--labeled-only", action="store_true", help="Process only labeled images")
    parser.add_argument("--unlabeled-only", action="store_true", help="Process only unlabeled images")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(CONFIG["log_dir"], f"inference_{timestamp}.log")
    logger = setup_logging(log_file)
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    print("="*80)
    print("nnU-Net 5-Fold Ensemble Inference with Detailed Progress")
    print("="*80)
    print(f"Configuration:")
    print(f"  Architecture: ResEnc{CONFIG['resenc_variant']}")
    print(f"  Dataset: {CONFIG['dataset_name']}")
    print(f"  Trainer: {CONFIG['trainer_name']}")
    print(f"  Folds: {CONFIG['folds']}")
    print(f"  Output: 3-class segmentation (0=background, 1=tumor, 2=pancreas)")
    print("="*80)
    
    try:
        # Set environment variables
        env_vars = set_environment_variables()
        logger.info("\nEnvironment variables set successfully")
        
        # Find model directory
        model_dir = find_model_directory()
        logger.info(f"Model directory: {model_dir}")
        
        # Verify checkpoints
        checkpoint_info = verify_checkpoints(model_dir, logger)
        
        # Process labeled images
        if not args.unlabeled_only:
            print("\n" + "="*60)
            print("PROCESSING LABELED IMAGES")
            print("="*60)
            
            labeled_images = get_image_list(CONFIG["labeled_images_dir"])
            logger.info(f"Found {len(labeled_images)} labeled images in {CONFIG['labeled_images_dir']}")
            
            if labeled_images:
                count = process_image_set(
                    labeled_images, 
                    CONFIG["output_labeled_dir"], 
                    model_dir, 
                    logger,
                    "labeled_images"
                )
                logger.info(f"\nLabeled images complete: {count} images → {CONFIG['output_labeled_dir']}")
        
        # Process unlabeled images
        if not args.labeled_only:
            print("\n" + "="*60)
            print("PROCESSING UNLABELED IMAGES")
            print("="*60)
            
            unlabeled_images = get_image_list(CONFIG["unlabeled_images_dir"])
            logger.info(f"Found {len(unlabeled_images)} unlabeled images in {CONFIG['unlabeled_images_dir']}")
            
            if unlabeled_images:
                count = process_image_set(
                    unlabeled_images, 
                    CONFIG["output_unlabeled_dir"], 
                    model_dir, 
                    logger,
                    "unlabeled_images"
                )
                logger.info(f"\nUnlabeled images complete: {count} images → {CONFIG['output_unlabeled_dir']}")
        
        # Final summary
        print("\n" + "="*80)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("="*80)
        print("Output locations:")
        print(f"  Labeled predictions: {CONFIG['output_labeled_dir']}")
        print(f"  Unlabeled predictions: {CONFIG['output_unlabeled_dir']}")
        print(f"  Log file: {log_file}")
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nERROR: {e}")
        raise

if __name__ == "__main__":
    main()