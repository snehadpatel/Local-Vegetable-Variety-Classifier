import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np

from models import (
    CustomGAPCNN,
    CustomMultiScaleCNN,
    CustomResidualCNN,
    CustomDeepCNN
)

CLASSES = ['Green Chilli (Marcha)', 'Ladies finger', 'Pointed gourd', 'ivy guard', 'peas']
PROCESSED_DATA_DIR = "dataset_processed"

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device initialized for live inference: {device}")
    
    # 1. Load Ensemble Models
    print("Loading models for Softmax Ensemble...")
    model_gap = CustomGAPCNN(num_classes=5)
    model_residual = CustomResidualCNN(num_classes=5)
    model_multiscale = CustomMultiScaleCNN(num_classes=5)
    model_deep = CustomDeepCNN(num_classes=5)
    
    models_dict = {
        "GAPCNN": (model_gap, "gap_model.pth", 0.26),
        "ResidualCNN": (model_residual, "residual_model.pth", 0.24),
        "MultiScaleCNN": (model_multiscale, "multiscale_model.pth", 0.22),
        "DeepCNN": (model_deep, "deep_model.pth", 0.28)
    }
    
    active_ensemble = []
    for name, (model, path, weight) in models_dict.items():
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            model.eval()
            active_ensemble.append((model, weight, name))
            print(f" -> Loaded {name} from '{path}' (Voting Weight: {weight:.2f})")
    
    if not active_ensemble:
        print("Error: No models could be loaded for ensemble!")
        return

    # 2. Prep transforms
    predict_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Find sample image from each class in the test set
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test")
    
    print("\n" + "="*80)
    print("                      LIVE ENSEMBLE INFERENCE TEST RUN")
    print("="*80)
    
    console_out = []
    
    header = f"{'True Class':<25} | {'Image Filename':<20} | {'Predicted Class':<25} | {'Confidence':<10}"
    divider = "-" * len(header)
    print(header)
    print(divider)
    console_out.append(header)
    console_out.append(divider)
    
    for cls in CLASSES:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.exists(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            continue
            
        # Select the first sample
        sample_img_name = images[0]
        img_path = os.path.join(cls_dir, sample_img_name)
        
        try:
            # Load and transform image
            image = Image.open(img_path).convert("RGB")
            tensor = predict_transforms(image).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                total_weight = sum(w for _, w, _ in active_ensemble)
                weighted_probs = np.zeros(len(CLASSES))
                
                for model, weight, name in active_ensemble:
                    outputs = model(tensor)
                    outputs = outputs / 0.8  # Temperature scaling
                    probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()
                    weighted_probs += probs * (weight / total_weight)
                
                # Apply Ivy Gourd penalty
                ivy_idx = CLASSES.index('ivy guard')
                weighted_probs[ivy_idx] *= 0.85
                
                # Re-normalize
                probabilities = weighted_probs / weighted_probs.sum()
                
            prediction_idx = probabilities.argmax()
            pred_class = CLASSES[prediction_idx]
            confidence = probabilities[prediction_idx]
            
            row = f"{cls:<25} | {sample_img_name:<20} | {pred_class:<25} | {confidence*100:.2f}%"
            print(row)
            console_out.append(row)
            
            # Print class-by-class probabilities breakdown
            prob_breakdown = "   Probabilities: " + ", ".join([f"{CLASSES[i]}: {probabilities[i]*100:.1f}%" for i in range(len(CLASSES))])
            print(prob_breakdown)
            console_out.append(prob_breakdown)
            console_out.append("")
            
        except Exception as e:
            print(f"Error predicting {img_path}: {e}")
            
    print("="*80)
    console_out.append("="*80)
    
    # Save printout to a text file
    with open("inference_console_output.txt", "w") as f:
        f.write("\n".join(console_out))
    print("Saved live inference console output to inference_console_output.txt")

if __name__ == "__main__":
    main()
