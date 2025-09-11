import os
from pathlib import Path
import json

def scan_mha_files(base_dir="./data/PANTHER_Task1"):
    """Scan for all .mha files in the dataset directories"""
    
    results = {
        "ImagesTr": [],
        "LabelsTr": [],
        "ImagesTr_unlabeled": []
    }
    
    # Scan ImagesTr (labeled images)
    images_tr_path = Path(base_dir) / "ImagesTr"
    if images_tr_path.exists():
        for file in sorted(images_tr_path.glob("*.mha")):
            results["ImagesTr"].append(file.name)
        print(f"Found {len(results['ImagesTr'])} files in ImagesTr")
    else:
        print(f"WARNING: {images_tr_path} does not exist")
    
    # Scan LabelsTr (labels)
    labels_tr_path = Path(base_dir) / "LabelsTr"
    if labels_tr_path.exists():
        for file in sorted(labels_tr_path.glob("*.mha")):
            results["LabelsTr"].append(file.name)
        print(f"Found {len(results['LabelsTr'])} files in LabelsTr")
    else:
        print(f"WARNING: {labels_tr_path} does not exist")
    
    # Scan ImagesTr_unlabeled (unlabeled images)
    images_unlabeled_path = Path(base_dir) / "ImagesTr_unlabeled"
    if images_unlabeled_path.exists():
        for file in sorted(images_unlabeled_path.glob("*.mha")):
            results["ImagesTr_unlabeled"].append(file.name)
        print(f"Found {len(results['ImagesTr_unlabeled'])} files in ImagesTr_unlabeled")
    else:
        print(f"WARNING: {images_unlabeled_path} does not exist")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"Labeled images (ImagesTr): {len(results['ImagesTr'])}")
    print(f"Labels (LabelsTr): {len(results['LabelsTr'])}")
    print(f"Unlabeled images (ImagesTr_unlabeled): {len(results['ImagesTr_unlabeled'])}")
    print(f"Total images: {len(results['ImagesTr']) + len(results['ImagesTr_unlabeled'])}")
    
    # Save to JSON for easy reference
    with open("mha_file_scan_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: mha_file_scan_results.json")
    
    # Print first few examples from each directory
    print("\n" + "="*60)
    print("SAMPLE FILES:")
    print("="*60)
    
    print("\nImagesTr (first 5):")
    for file in results["ImagesTr"][:5]:
        print(f"  {file}")
    
    print("\nLabelsTr (first 5):")
    for file in results["LabelsTr"][:5]:
        print(f"  {file}")
    
    print("\nImagesTr_unlabeled (first 5):")
    for file in results["ImagesTr_unlabeled"][:5]:
        print(f"  {file}")
    
    return results

if __name__ == "__main__":
    # Run the scan
    results = scan_mha_files()
