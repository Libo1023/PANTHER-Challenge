import os
import sys
import subprocess
import json
import shutil
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from datetime import datetime
import logging
from scipy import stats
import glob
import warnings
warnings.filterwarnings("ignore")

class PancreaticTumorSegmentationContainer:
    def __init__(self):
        
        # Configuration
        self.num_folds = 5 
        self.dataset_id = 91
        self.dataset_name = "Dataset091_PantherStudent"
        self.trainer_name = "nnUNetTrainer_Student"
        self.plans_name = "nnUNetResEncUNetMPlans"
        self.ensemble_method = "mean"  # or "majority_vote"
        
        # nnUNet paths
        self.nnunet_input_dir = Path("/opt/algorithm/nnunet/input")
        self.nnunet_output_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_model_dir = Path("/opt/algorithm/nnunet/nnUNet_results")
        self.temp_dir = Path("/opt/algorithm/temp")
        
        # Input/output paths for the challenge
        folders_with_mri = [folder for folder in os.listdir("/input/images") if "mri" in folder.lower()]
        if len(folders_with_mri) == 1:
            mr_ip_dir_name = folders_with_mri[0]
            print(f"Found MRI folder: {mr_ip_dir_name}")
        else:
            print(f"Warning: Expected one folder containing 'mri', but found {len(folders_with_mri)}")
            mr_ip_dir_name = 'abdominal-t1-mri'  # default value
        
        self.mr_ip_dir = Path(f"/input/images/{mr_ip_dir_name}")
        self.output_dir = Path("/output")
        self.output_dir_images = Path(os.path.join(self.output_dir, "images"))
        self.output_dir_seg_mask = Path(os.path.join(self.output_dir_images, "pancreatic-tumor-segmentation"))
        self.segmentation_mask = self.output_dir_seg_mask / "tumor_seg.mha"
        
        # Model weights path (if uploaded separately)
        self.weights_path = Path("/opt/ml/model")
        
        # Ensure required folders exist
        self.nnunet_input_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir_seg_mask.mkdir(exist_ok=True, parents=True)
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Find input image
        mha_files = glob.glob(os.path.join(self.mr_ip_dir, '*.mha'))
        if mha_files:
            self.mr_image = mha_files[0]
            print(f"Found input image: {self.mr_image}")
        else:
            raise RuntimeError('No .mha images found in input directory')
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.temp_dir / f"inference_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    def setup_nnunet_environment(self, logger):
        """Setup nnU-Net environment and model checkpoints"""
        # Add custom trainer path to Python path
        trainer_dir = "/opt/algorithm/custom_trainers"
        if trainer_dir not in sys.path:
            sys.path.insert(0, trainer_dir)
            logger.info(f"Added custom trainer path: {trainer_dir}")
        
        # If model weights are uploaded separately, move them to proper location
        if self.weights_path.exists():
            logger.info("Found uploaded model weights, moving to nnUNet structure...")
            self.move_checkpoints(self.weights_path, logger)
        else:
            logger.info("Using model weights from container image")
            # Verify checkpoints exist in the expected location
            for fold in range(self.num_folds):
                checkpoint_path = self.nnunet_model_dir / self.dataset_name / \
                    f"{self.trainer_name}__{self.plans_name}__3d_fullres" / \
                    f"fold_{fold}" / "checkpoint_best.pth"
                if not checkpoint_path.exists():
                    logger.warning(f"Checkpoint not found at: {checkpoint_path}")
                    # Try without .pth extension
                    checkpoint_path_no_ext = checkpoint_path.parent / "checkpoint_best"
                    if checkpoint_path_no_ext.exists():
                        logger.info(f"Found checkpoint without extension for fold {fold}")
                    else:
                        raise RuntimeError(f"Checkpoint not found for fold {fold}")
        
        return True
    
    def move_checkpoints(self, source_dir, logger):
        """Move nnUNet checkpoints to nnUNet_results directory"""
        task_name = self.dataset_name.split("_")[1]
        
        # Create the target directory structure
        target_base = self.nnunet_model_dir / self.dataset_name / \
            f"{self.trainer_name}__{self.plans_name}__3d_fullres"
        
        for fold in range(self.num_folds):
            # Create fold directory
            fold_dir = target_base / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            
            # Look for checkpoint file
            source_path = source_dir / f"checkpoint_best_{task_name}_fold_{fold}.pth"
            if source_path.exists():
                # Copy and rename to checkpoint_best (without .pth for nnUNet)
                destination_path = fold_dir / "checkpoint_best"
                shutil.copyfile(source_path, destination_path)
                logger.info(f"Moved checkpoint for fold {fold}")
            else:
                logger.error(f"Source checkpoint not found: {source_path}")
    
    def run_fold_inference(self, input_image_path, fold, logger):
        """Run inference for a single fold"""
        # Prepare input directory for this fold
        fold_input_dir = self.temp_dir / "fold_inputs" / f"fold_{fold}"
        fold_input_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy input image with _0000 suffix for nnUNet
        input_name = Path(input_image_path).stem
        shutil.copy2(input_image_path, fold_input_dir / f"{input_name}_0000.mha")
        
        # Output directory for this fold
        fold_output_dir = self.temp_dir / "predictions_per_fold" / f"fold_{fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing predictions
        for f in fold_output_dir.glob("*.nii.gz"):
            f.unlink()
        for f in fold_output_dir.glob("*.mha"):
            f.unlink()
        
        # Build inference command
        cmd = [
            "nnUNetv2_predict",
            "-i", str(fold_input_dir),
            "-o", str(fold_output_dir),
            "-d", str(self.dataset_id),
            "-p", self.plans_name,
            "-c", "3d_fullres",
            "-f", str(fold),
            "-tr", self.trainer_name,
            "-chk", "checkpoint_best",
            "--disable_progress_bar"
        ]
        
        logger.info(f"Running inference for fold {fold}...")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            # Run inference
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Inference failed for fold {fold}")
                logger.error(f"stdout: {result.stdout}")
                logger.error(f"stderr: {result.stderr}")
                return None
            
            # Find output file
            output_files = list(fold_output_dir.glob("*.nii.gz"))
            if len(output_files) == 0:
                output_files = list(fold_output_dir.glob("*.mha"))
            
            if len(output_files) == 0:
                logger.error(f"No output file generated for fold {fold}")
                return None
            
            logger.info(f"Inference completed for fold {fold}")
            return output_files[0]
        
        except Exception as e:
            logger.error(f"Exception during inference for fold {fold}: {str(e)}")
            return None
        finally:
            # Clean up fold input directory
            shutil.rmtree(fold_input_dir, ignore_errors=True)
    
    def ensemble_predictions(self, prediction_paths, output_path, logger):
        """Ensemble multiple predictions and extract tumor class"""
        logger.info(f"Ensembling {len(prediction_paths)} predictions...")
        
        # Load all predictions
        predictions = []
        reference_image = None
        
        for i, pred_path in enumerate(prediction_paths):
            try:
                img = sitk.ReadImage(str(pred_path))
                arr = sitk.GetArrayFromImage(img)
                predictions.append(arr)
                
                if i == 0:
                    reference_image = img
                
                # Log unique values
                unique_vals = np.unique(arr)
                logger.info(f"  Fold {i} unique values: {unique_vals}")
                
                # Count voxels per class
                for val in unique_vals:
                    count = np.sum(arr == val)
                    logger.info(f"    Class {int(val)}: {count} voxels")
                    
            except Exception as e:
                logger.error(f"Failed to load prediction from fold {i}: {e}")
                return False
        
        # Stack predictions
        predictions = np.stack(predictions, axis=0)  # Shape: (n_folds, z, y, x)
        
        # Ensemble method
        if self.ensemble_method == "mean":
            # Soft voting via one-hot encoding
            n_classes = 3
            one_hot = np.zeros((*predictions.shape, n_classes))
            
            for c in range(n_classes):
                one_hot[..., c] = (predictions == c).astype(float)
            
            # Average across folds
            mean_probs = np.mean(one_hot, axis=0)  # Shape: (z, y, x, n_classes)
            
            # Get class with highest probability
            ensemble_multiclass = np.argmax(mean_probs, axis=-1)
            
        else:  # majority_vote
            # Use scipy mode for majority voting
            ensemble_multiclass, _ = stats.mode(predictions, axis=0, keepdims=False)
            if isinstance(ensemble_multiclass, np.ndarray) and ensemble_multiclass.ndim > 3:
                ensemble_multiclass = ensemble_multiclass.squeeze()
        
        # Extract tumor class (class 1) for binary mask
        binary_mask = (ensemble_multiclass == 1).astype(np.uint8)
        
        # Log results
        logger.info(f"Ensemble results:")
        logger.info(f"  Multi-class unique values: {np.unique(ensemble_multiclass)}")
        logger.info(f"  Binary mask unique values: {np.unique(binary_mask)}")
        logger.info(f"  Total tumor voxels: {np.sum(binary_mask)}")
        
        # Calculate statistics
        total_voxels = binary_mask.size
        tumor_percentage = (np.sum(binary_mask) / total_voxels) * 100
        logger.info(f"  Tumor percentage: {tumor_percentage:.2f}%")
        
        # Create final binary output
        output_img = sitk.GetImageFromArray(binary_mask)
        output_img.CopyInformation(reference_image)
        sitk.WriteImage(output_img, str(output_path))
        logger.info(f"Saved binary segmentation to: {output_path}")
        
        return True
    
    def run(self):
        """Main inference pipeline"""
        _show_torch_cuda_info()
        
        # Setup logging
        logger = self.setup_logging()
        
        logger.info("="*60)
        logger.info("PANTHER 5-FOLD ENSEMBLE INFERENCE")
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Trainer: {self.trainer_name}")
        logger.info(f"Plans: {self.plans_name}")
        logger.info(f"Ensemble method: {self.ensemble_method}")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        # Setup nnU-Net environment
        logger.info("\n1. Setting up nnU-Net environment...")
        if not self.setup_nnunet_environment(logger):
            logger.error("Failed to setup nnU-Net environment")
            return
        logger.info("nnU-Net environment setup complete")
        
        # Load and log input image info
        logger.info("\n2. Processing input image...")
        try:
            itk_image = sitk.ReadImage(self.mr_image)
            logger.info(f"Input image: {os.path.basename(self.mr_image)}")
            logger.info(f"  Shape: {itk_image.GetSize()}")
            logger.info(f"  Spacing: {itk_image.GetSpacing()}")
            logger.info(f"  Origin: {itk_image.GetOrigin()}")
        except Exception as e:
            logger.error(f"Failed to read input image: {e}")
            return
        
        # Run inference for each fold
        logger.info("\n3. Running inference for each fold...")
        prediction_paths = []
        
        for fold in range(self.num_folds):
            logger.info(f"\nProcessing fold {fold}...")
            pred_path = self.run_fold_inference(self.mr_image, fold, logger)
            if pred_path is None:
                logger.error(f"Failed to generate prediction for fold {fold}")
                return
            prediction_paths.append(pred_path)
            logger.info(f"Fold {fold} complete")
        
        # Ensemble predictions
        logger.info("\n4. Ensembling predictions...")
        if not self.ensemble_predictions(prediction_paths, self.segmentation_mask, logger):
            logger.error("Ensemble failed")
            return
        
        # Calculate total time
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*60)
        logger.info("INFERENCE COMPLETED SUCCESSFULLY")
        logger.info(f"Total inference time: {total_time:.1f} seconds")
        logger.info(f"Output saved to: {self.segmentation_mask}")
        logger.info("="*60)

def _show_torch_cuda_info():
    """Display PyTorch CUDA information"""
    import torch
    
    print("=+=" * 10)
    print(f"PyTorch version: {torch.__version__}")
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    
    if available:
        print(f"\tNumber of devices: {torch.cuda.device_count()}")
        print(f"\tCurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tDevice properties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

if __name__ == "__main__":
    PancreaticTumorSegmentationContainer().run()