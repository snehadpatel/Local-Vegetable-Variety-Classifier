import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import time
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Import utilities
from utils import TargetedAugmentationDataset, EarlyStopping, get_class_weights

# Configurations
PROCESSED_DATA_DIR = "dataset_processed"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Hardware Acceleration: Detected Apple Silicon GPU! Training on 'mps'.")
    else:
        device = torch.device("cpu")
        print("Hardware Acceleration: Active CPU.")

    # 1. Dataset & DataLoaders
    train_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
    val_dir = os.path.join(PROCESSED_DATA_DIR, 'val')
    test_dir = os.path.join(PROCESSED_DATA_DIR, 'test')

    standard_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ivy_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = TargetedAugmentationDataset(train_dir, standard_transform=standard_transform, ivy_transform=ivy_transform)
        val_dataset = TargetedAugmentationDataset(val_dir, standard_transform=standard_transform, ivy_transform=None)
        test_dataset = TargetedAugmentationDataset(test_dir, standard_transform=standard_transform, ivy_transform=None)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure the dataset is perfectly balanced and exists.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    classes = train_dataset.classes
    print(f"Loaded {len(train_dataset)} training images.")
    print(f"Loaded {len(val_dataset)} validation images.")
    print(f"Loaded {len(test_dataset)} test images.")
    print(f"Detected classes: {classes}")

    class_weights_tensor = get_class_weights(train_dataset).to(device)

    print("\n" + "="*50)
    print("           TRAINING PRE-TRAINED RESNET-18")
    print("="*50)

    # 2. Load Pre-trained ResNet-18
    # Using the new weights parameter instead of pretrained=True
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze the base layers to act as a pure feature extractor
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    # We unfreeze the final layer so it learns our specific 5 classes
    model.fc = nn.Linear(num_ftrs, len(classes))
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # Only optimize the parameters that require gradients (the final fc layer)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=3, delta=0.01, verbose=True, path="resnet18_model.pth")

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct / total

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct / total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            break

    print(f"\nTraining completed! Best Validation Loss: {early_stopping.val_loss_min}")

    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss vs Epochs (ResNet-18)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy vs Epochs (ResNet-18)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('resnet18_curves.png')
    print("Saved learning curves as 'resnet18_curves.png'.")

    # 3. Test Evaluation
    print("\n" + "="*50)
    print("           EVALUATION ON UNSEEN TEST SET")
    print("="*50)
    model.load_state_dict(torch.load("resnet18_model.pth", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("="*50)

if __name__ == '__main__':
    main()
