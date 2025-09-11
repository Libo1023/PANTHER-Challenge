#!/usr/bin/env python3
"""
Script to evaluate 3-class segmentations by converting them to binary (tumor-only)
and then calling evaluate_local.py
"""

import os
import shutil
import tempfile
import SimpleITK as sitk
from pathlib import Path
import sys

# Import the evaluation function directly
from evaluate_local import evaluate_segmentation_performance

def convert_3class_to_binary(input_path, output_path):
    """
    Convert 3-class segmentation to binary:
    - Label 0 (background) → 0
    - Label 1 (tumor) → 1
    - Label 2 (pancreas) → 0
    """
    # Read the image
    image = sitk.ReadImage(input_path)
    array = sitk.GetArrayFromImage(image)
    
    # Convert: keep only label 1 as 1, everything else becomes 0
    binary_array = (array == 1).astype('uint8')
    
    # Create new image with same metadata
    binary_image = sitk.GetImageFromArray(binary_array)
    binary_image.CopyInformation(image)
    
    # Save the binary image
    sitk.WriteImage(binary_image, output_path)

def main():
    # Define directories
    gt_dir = "./data/PANTHER_Task1/LabelsTr/"
    pred_dir = "./data/PANTHER_Task1/PredictionsTr_3class/"
    pred_dir = "./fold_2_predictions/"
    
    # Check if directories exist
    if not os.path.exists(gt_dir):
        print(f"Error: Ground truth directory not found: {gt_dir}")
        sys.exit(1)
    if not os.path.exists(pred_dir):
        print(f"Error: Prediction directory not found: {pred_dir}")
        sys.exit(1)
    
    # Get all .mha files from both directories
    gt_files = {f for f in os.listdir(gt_dir) if f.endswith('.mha')}
    pred_files = {f for f in os.listdir(pred_dir) if f.endswith('.mha')}
    
    # Find matching files
    matching_files = gt_files.intersection(pred_files)
    
    # Report findings
    print("=" * 60)
    print("FILE MAPPING REPORT")
    print("=" * 60)
    print(f"Ground truth directory: {gt_dir}")
    print(f"Found {len(gt_files)} .mha files")
    print(f"\nPrediction directory: {pred_dir}")
    print(f"Found {len(pred_files)} .mha files")
    print(f"\nMatching files: {len(matching_files)}")
    
    if len(matching_files) == 0:
        print("\nError: No matching files found!")
        sys.exit(1)
    
    print("\nMatching files:")
    for i, fname in enumerate(sorted(matching_files), 1):
        print(f"  {i}. {fname}")
    
    # Files only in GT
    gt_only = gt_files - pred_files
    if gt_only:
        print(f"\nFiles only in GT directory ({len(gt_only)}):")
        for fname in sorted(gt_only):
            print(f"  - {fname}")
    
    # Files only in predictions
    pred_only = pred_files - gt_files
    if pred_only:
        print(f"\nFiles only in predictions directory ({len(pred_only)}):")
        for fname in sorted(pred_only):
            print(f"  - {fname}")
    
    print("\n" + "=" * 60)
    print("CONVERTING 3-CLASS TO BINARY SEGMENTATIONS")
    print("=" * 60)
    
    # Create temporary directories for binary masks
    with tempfile.TemporaryDirectory() as temp_gt_dir, \
         tempfile.TemporaryDirectory() as temp_pred_dir:
        
        # Convert all matching files
        for i, fname in enumerate(sorted(matching_files), 1):
            print(f"Converting {i}/{len(matching_files)}: {fname}")
            
            # Convert GT (3-class to binary)
            gt_path = os.path.join(gt_dir, fname)
            temp_gt_path = os.path.join(temp_gt_dir, fname)
            convert_3class_to_binary(gt_path, temp_gt_path)
            
            # Convert predictions (3-class to binary)
            pred_path = os.path.join(pred_dir, fname)
            temp_pred_path = os.path.join(temp_pred_dir, fname)
            convert_3class_to_binary(pred_path, temp_pred_path)
        
        print("\nConversion complete!")
        
        # Create subject list (remove .mha extension)
        subject_list = [fname[:-4] for fname in sorted(matching_files)]
        
        print("\n" + "=" * 60)
        print("RUNNING EVALUATION")
        print("=" * 60)
        
        # Call the evaluation function directly
        print("Computing PANTHER evaluation metrics...")
        results = evaluate_segmentation_performance(
            pred_dir=temp_pred_dir,
            gt_dir=temp_gt_dir,
            subject_list=subject_list,
            verbose=True  # This will print per-subject metrics
        )
        
        # Display final aggregated metrics
        print("\n" + "=" * 60)
        print("FINAL AGGREGATED METRICS (TUMOR SEGMENTATION)")
        print("=" * 60)
        
        agg = results['aggregates']
        print(f"Mean Volumetric Dice:     {agg['mean_volumetric_dice']:.4f}")
        print(f"Mean Surface Dice (5mm):  {agg['mean_surface_dice']:.4f}")
        print(f"Mean Hausdorff95:         {agg['mean_hausdorff95']:.4f} mm")
        print(f"Mean MASD:                {agg['mean_masd']:.4f} mm")
        print(f"Tumor Burden RMSE:        {agg['tumor_burden_rmse']:.4f} mm³")
        
        # Also display per-subject summary
        print("\n" + "=" * 60)
        print("PER-SUBJECT SUMMARY")
        print("=" * 60)
        print(f"{'Subject':<20} {'Dice':>8} {'SurfDice':>10} {'HD95':>10} {'MASD':>10}")
        print("-" * 60)
        
        for m in results['per_subject']:
            print(f"{m['subject']:<20} "
                  f"{m['volumetric_dice']:>8.4f} "
                  f"{m['surface_dice']:>10.4f} "
                  f"{m['hausdorff95']:>10.2f} "
                  f"{m['masd']:>10.2f}")
        
        print("=" * 60)
        print("Evaluation complete!")

if __name__ == "__main__":
    main()
