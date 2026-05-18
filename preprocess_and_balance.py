import os
import shutil
import random
from PIL import Image, ImageEnhance

# Set seeds for reproducibility
random.seed(42)

# Configurations
RAW_DATA_DIR = "/Users/snehapatel/Library/CloudStorage/GoogleDrive-sneha.dipan.dec2005@gmail.com/My Drive/Local Vegetable Variety Classifier/dataset"
PROCESSED_DATA_DIR = "/Users/snehapatel/Library/CloudStorage/GoogleDrive-sneha.dipan.dec2005@gmail.com/My Drive/Local Vegetable Variety Classifier/dataset_processed"
IMG_SIZE = (224, 224)


def setup_dirs(classes):
    """Creates the target directory structure, deleting any existing preprocessed data."""
    if os.path.exists(PROCESSED_DATA_DIR):
        print(f"Removing old processed dataset directory: {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)
        
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(PROCESSED_DATA_DIR, split, cls), exist_ok=True)
    print("Created new directory structure for processed splits.")

def get_valid_files(class_dir):
    """Returns a list of valid image paths from a class directory, skipping hidden files."""
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}
    files = []
    for f in os.listdir(class_dir):
        if f.startswith('.'):
            continue
        ext = os.path.splitext(f)[1].lower()
        if ext in valid_exts:
            files.append(f)
    return files

def process_and_save_image(src_path, dest_path):
    """Loads, converts to RGB, resizes, and saves an image as JPEG."""
    try:
        with Image.open(src_path) as img:
            # Convert to RGB mode (handles PNG alpha/CMYK/grayscale)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # High-quality resize using LANCZOS filter
            img_resized = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            img_resized.save(dest_path, 'JPEG', quality=95)
            return True
    except Exception as e:
        print(f"Warning: Failed to process {src_path}. Error: {e}")
        return False



def main():
    if not os.path.exists(RAW_DATA_DIR):
        print(f"Error: Raw dataset path {RAW_DATA_DIR} not found.")
        return
        
    classes = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d)) and not d.startswith('.')]
    print(f"Detected {len(classes)} classes: {classes}")
    
    setup_dirs(classes)
    
    # Store stats to print at the end
    stats = {}
    
    for cls in classes:
        cls_src_dir = os.path.join(RAW_DATA_DIR, cls)
        all_files = get_valid_files(cls_src_dir)
        random.shuffle(all_files)  # Shuffle files randomly for partition split
        
        total_files = len(all_files)
        
        # Calculate split sizes (70% train, 15% val, 15% test)
        train_end = int(total_files * 0.70)
        val_end = train_end + int(total_files * 0.15)
        
        train_files = all_files[:train_end]
        val_files = all_files[train_end:val_end]
        # Rest goes to test to ensure we don't drop any leftover images due to rounding
        test_files = all_files[val_end:]
        
        stats[cls] = {
            "raw_total": total_files,
            "raw_train": len(train_files),
            "raw_val": len(val_files),
            "raw_test": len(test_files),
            "processed_train": 0,
            "processed_val": 0,
            "processed_test": 0
        }
        
        print(f"\n--- Processing class: '{cls}' ---")
        
        # 1. Process Validation Set (Standardize only)
        print(f"Processing Validation split ({len(val_files)} images)...")
        for f in val_files:
            src = os.path.join(cls_src_dir, f)
            dest = os.path.join(PROCESSED_DATA_DIR, "val", cls, f)
            if process_and_save_image(src, dest):
                stats[cls]["processed_val"] += 1
                
        # 2. Process Test Set (Standardize only)
        print(f"Processing Test split ({len(test_files)} images)...")
        for f in test_files:
            src = os.path.join(cls_src_dir, f)
            dest = os.path.join(PROCESSED_DATA_DIR, "test", cls, f)
            if process_and_save_image(src, dest):
                stats[cls]["processed_test"] += 1
                
        # 3. Process Train Set (Standardize first)
        print(f"Processing Training split ({len(train_files)} images)...")
        saved_train_paths = []
        for f in train_files:
            src = os.path.join(cls_src_dir, f)
            # Standardize filename format to ensure consistency
            base_name, _ = os.path.splitext(f)
            dest_filename = f"{base_name}.jpg"
            dest = os.path.join(PROCESSED_DATA_DIR, "train", cls, dest_filename)
            
            if process_and_save_image(src, dest):
                stats[cls]["processed_train"] += 1
                saved_train_paths.append(dest)
                

            
    # Print beautiful summary report
    print("\n" + "="*80)
    print("                           DATASET BALANCING REPORT")
    print("="*80)
    print(f"{'Class':<25} | {'Raw Total':<10} | {'Train (Clean)':<15} | {'Val (Clean)':<12} | {'Test (Clean)':<12}")
    print("-"*80)
    for cls, data in stats.items():
        print(f"{cls:<25} | {data['raw_total']:<10} | {data['processed_train']:<15} | {data['processed_val']:<12} | {data['processed_test']:<12}")
    print("="*80)
    print("Dataset splitting completed successfully (relies on real-time PyTorch augmentation)!")

if __name__ == '__main__':
    main()
