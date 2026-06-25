import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import json

from models import (
    CustomLightCNN,
    CustomDeepCNN,
    CustomGAPCNN,
    CustomMultiScaleCNN,
    CustomResidualCNN,
    CustomDepthwiseSepCNN
)

PROCESSED_DATA_DIR = "dataset_processed"
BATCH_SIZE = 32

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up test dataloader
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test")
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    classes = test_dataset.classes
    num_classes = len(classes)
    print(f"Classes: {classes}")
    print(f"Loaded {len(test_dataset)} test images.")

    # Dictionary of models and their classes
    model_configs = {
        "LightCNN": {
            "class": CustomLightCNN,
            "path": "light_model.pth",
            "type": "custom"
        },
        "DeepCNN": {
            "class": CustomDeepCNN,
            "path": "deep_model.pth",
            "type": "custom"
        },
        "GAPCNN": {
            "class": CustomGAPCNN,
            "path": "gap_model.pth",
            "type": "custom"
        },
        "MultiScaleCNN": {
            "class": CustomMultiScaleCNN,
            "path": "multiscale_model.pth",
            "type": "custom"
        },
        "ResidualCNN": {
            "class": CustomResidualCNN,
            "path": "residual_model.pth",
            "type": "custom"
        },
        "DepthwiseSepCNN": {
            "class": CustomDepthwiseSepCNN,
            "path": "depthwise_model.pth",
            "type": "custom"
        },
        "ResNet18": {
            "type": "resnet18",
            "path": "resnet18_model.pth"
        }
    }

    results = {}

    for name, config in model_configs.items():
        print(f"\nEvaluating {name}...")
        path = config["path"]
        if not os.path.exists(path):
            print(f"Checkpoint not found for {name}: {path}")
            continue

        # Instantiate model
        if config["type"] == "custom":
            model = config["class"](num_classes=num_classes)
        elif config["type"] == "resnet18":
            model = models.resnet18()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Generate report and confusion matrix
        report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds).tolist()

        # Print quick accuracy
        acc = report["accuracy"]
        print(f"{name} Test Accuracy: {acc*100:.2f}%")

        results[name] = {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm
        }

    # Save to JSON
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nSaved evaluation results to evaluation_results.json")

if __name__ == "__main__":
    main()
