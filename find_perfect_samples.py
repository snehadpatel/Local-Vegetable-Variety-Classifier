import os
import torch
from torchvision import transforms
from PIL import Image
import shutil
import glob
from models import CustomGAPCNN, CustomMultiScaleCNN, CustomResidualCNN, CustomDeepCNN

def main():
    # Setup device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    CLASSES = ['Green Chilli (Marcha)', 'Ladies finger', 'Pointed gourd', 'ivy guard', 'peas']

    print("Loading models...")
    # Initialize models
    gap_model = CustomGAPCNN(num_classes=5).to(device)
    multiscale_model = CustomMultiScaleCNN(num_classes=5).to(device)
    residual_model = CustomResidualCNN(num_classes=5).to(device)
    deep_model = CustomDeepCNN(num_classes=5).to(device)

    # Load the NEW weights!
    gap_model.load_state_dict(torch.load("gap_model.pth", map_location=device))
    multiscale_model.load_state_dict(torch.load("multiscale_model.pth", map_location=device))
    residual_model.load_state_dict(torch.load("residual_model.pth", map_location=device))
    deep_model.load_state_dict(torch.load("deep_model.pth", map_location=device))

    models = [gap_model, multiscale_model, residual_model, deep_model]
    for m in models: m.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def get_ensemble_prediction(img_path):
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            probs = []
            for m in models:
                outputs = m(input_tensor)
                prob = torch.nn.functional.softmax(outputs, dim=1)
                probs.append(prob)
                
            avg_probs = torch.mean(torch.stack(probs), dim=0).squeeze()
            
            # Bayesian prior calibration as in app.py
            ivy_idx = CLASSES.index('ivy guard')
            avg_probs[ivy_idx] *= 0.15
            avg_probs = avg_probs / avg_probs.sum()
            
            predicted_class = torch.argmax(avg_probs).item()
            confidence = avg_probs[predicted_class].item()
            
            return CLASSES[predicted_class], confidence

    os.makedirs("static/samples", exist_ok=True)
    file_mapping = {
        'Green Chilli (Marcha)': 'green_chilli.jpg',
        'Ladies finger': 'ladies_finger.jpg',
        'Pointed gourd': 'pointed_gourd.jpg',
        'ivy guard': 'ivy_guard.jpg',
        'peas': 'peas.jpg'
    }
    
    for cls in CLASSES:
        print(f"Searching for perfect image for: {cls}")
        best_img = None
        best_conf = 0.0
        
        # Search the test set
        img_paths = glob.glob(f"dataset_processed/test/{cls}/*.jpg")
        for p in img_paths:
            try:
                pred_cls, conf = get_ensemble_prediction(p)
                if pred_cls == cls and conf > best_conf:
                    best_conf = conf
                    best_img = p
            except Exception as e:
                pass
                
        if best_img:
            print(f"✅ Found {best_img} with confidence {best_conf:.4f}")
            # Ensure it is a fresh copy
            shutil.copy(best_img, os.path.join("static/samples", file_mapping[cls]))
        else:
            print(f"❌ Could not find a high-confidence working image for {cls} in the test set!")

if __name__ == "__main__":
    main()
