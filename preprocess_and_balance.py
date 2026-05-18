import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageOps

# Set seeds for reproducibility
random.seed(42)

# Configurations
RAW_DATA_DIR = "dataset"
PROCESSED_DATA_DIR = "dataset_processed"
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

def augment_image(img):
    """Applies random transformations to a PIL Image."""
    # Random flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
    if random.random() > 0.5:
        img = ImageOps.flip(img)
        
    # Random rotation (-15 to 15 degrees)
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(255,255,255))
    
    # Random brightness (0.8 to 1.2)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # Random contrast (0.8 to 1.2)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    return img

def process_and_save_image(src_path, dest_path, augment=False):
    """Loads, converts to RGB, optionally augments, resizes, and saves an image as JPEG."""
    try:
        with Image.open(src_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            if augment:
                img = augment_image(img)
                
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
    class_files = {}
    
    for cls in classes:
        cls_src_dir = os.path.join(RAW_DATA_DIR, cls)
        all_files = get_valid_files(cls_src_dir)
        random.shuffle(all_files)
        class_files[cls] = all_files
        
    # Find the maximum number of images in any class to balance against
    max_total = max([len(files) for files in class_files.values()])
    max_train_size = int(max_total * 0.70)
    
    print(f"Balancing dataset... Target train size per class: {max_train_size}")
    
    for cls in classes:
        cls_src_dir = os.path.join(RAW_DATA_DIR, cls)
        all_files = class_files[cls]
        total_files = len(all_files)
        
        train_end = int(total_files * 0.70)
        val_end = train_end + int(total_files * 0.15)
        
        train_files = all_files[:train_end]
        val_files = all_files[train_end:val_end]
        test_files = all_files[val_end:]
        
        stats[cls] = {
            "raw_total": total_files,
            "processed_train": 0,
            "processed_val": 0,
            "processed_test": 0
        }
        
        print(f"\n--- Processing class: '{cls}' ---")
        
        # 1. Process Validation Set
        for f in val_files:
            src = os.path.join(cls_src_dir, f)
            dest = os.path.join(PROCESSED_DATA_DIR, "val", cls, f)
            if process_and_save_image(src, dest):
                stats[cls]["processed_val"] += 1
                
        # 2. Process Test Set
        for f in test_files:
            src = os.path.join(cls_src_dir, f)
            dest = os.path.join(PROCESSED_DATA_DIR, "test", cls, f)
            if process_and_save_image(src, dest):
                stats[cls]["processed_test"] += 1
                
        # 3. Process Train Set (Originals)
        for f in train_files:
            src = os.path.join(cls_src_dir, f)
            base_name, _ = os.path.splitext(f)
            dest = os.path.join(PROCESSED_DATA_DIR, "train", cls, f"{base_name}.jpg")
            if process_and_save_image(src, dest):
                stats[cls]["processed_train"] += 1
                
        # 4. Augment Train Set to match max_train_size
        deficit = max_train_size - len(train_files)
        if deficit > 0:
            print(f"Generating {deficit} augmented images for {cls}...")
            for i in range(deficit):
                src_file = random.choice(train_files)
                src = os.path.join(cls_src_dir, src_file)
                base_name, _ = os.path.splitext(src_file)
                dest = os.path.join(PROCESSED_DATA_DIR, "train", cls, f"{base_name}_aug_{i}.jpg")
                if process_and_save_image(src, dest, augment=True):
                    stats[cls]["processed_train"] += 1

    # Print beautiful summary report
    print("\n" + "="*80)
    print("                           DATASET BALANCING REPORT")
    print("="*80)
    print(f"{'Class':<25} | {'Raw Total':<10} | {'Train (Clean)':<15} | {'Val (Clean)':<12} | {'Test (Clean)':<12}")
    print("-"*80)
    for cls, data in stats.items():
        print(f"{cls:<25} | {data['raw_total']:<10} | {data['processed_train']:<15} | {data['processed_val']:<12} | {data['processed_test']:<12}")
    print("="*80)
    print("Dataset splitting and OFFLINE AUGMENTATION completed successfully!")

if __name__ == '__main__':
    main()
