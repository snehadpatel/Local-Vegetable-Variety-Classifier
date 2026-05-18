import os
import io
import multiprocessing
from pathlib import Path
from rembg import remove
from PIL import Image

def process_single_image(args):
    img_path, out_path = args
    if out_path.exists():
        return True # Already processed
        
    try:
        with open(img_path, "rb") as f:
            input_bytes = f.read()
            
        output_bytes = remove(input_bytes)
        
        img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        black_bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        black_bg.paste(img, (0, 0), img)
        
        final_img = black_bg.convert("RGB")
        final_img.save(out_path)
        return True
    except Exception as e:
        print(f"Failed to process {img_path.name}: {e}")
        return False

def process_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not output_path.exists():
        output_path.mkdir(parents=True)
        
    print(f"Starting FAST background removal from {input_dir} to {output_dir}")
    
    tasks = []
    
    for split_folder in input_path.iterdir():
        if not split_folder.is_dir(): continue
        for class_folder in split_folder.iterdir():
            if not class_folder.is_dir(): continue
                
            out_class_folder = output_path / split_folder.name / class_folder.name
            out_class_folder.mkdir(exist_ok=True, parents=True)
            
            for img_file in class_folder.iterdir():
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.webp']: continue
                out_file = out_class_folder / (img_file.stem + '.png')
                if not out_file.exists():
                    tasks.append((img_file, out_file))
    
    print(f"Found {len(tasks)} images remaining to process.")
    
    if len(tasks) > 0:
        # Use available CPU cores
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"Accelerating with {num_cores} parallel CPU cores!")
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(process_single_image, tasks)
            
        return sum(results)
    return 0

if __name__ == "__main__":
    total = process_dataset("dataset_processed", "dataset_nobg")
    print(f"Background removal complete! Total newly processed: {total}")
