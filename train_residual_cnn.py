import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from models import CustomResidualCNN
from utils import TargetedAugmentationDataset, EarlyStopping, get_class_weights

# Configurations
PROCESSED_DATA_DIR = "dataset_processed"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
if 'CustomResidualCNN' == 'CustomDeepCNN':
    EPOCHS = 8

def main():
    # 1. Device Setup (Leveraging hardware acceleration)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Hardware Acceleration: Detected Apple Silicon GPU! Training on 'mps'.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Hardware Acceleration: Detected NVIDIA GPU! Training on 'cuda'.")
    else:
        device = torch.device("cpu")
        print("Hardware Acceleration: No GPU detected. Training on 'cpu'.")
        
    # 2. Data Transforms with targeted Ivy Gourd Augmentation
    standard_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    ivy_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load Datasets using TargetedAugmentationDataset
    train_dir = os.path.join(PROCESSED_DATA_DIR, "train")
    val_dir = os.path.join(PROCESSED_DATA_DIR, "val")
    test_dir = os.path.join(PROCESSED_DATA_DIR, "test")
    
    train_dataset = TargetedAugmentationDataset(train_dir, standard_transform=standard_transform, ivy_transform=ivy_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    classes = train_dataset.classes
    print(f"Loaded {len(train_dataset)} training images (perfectly balanced).")
    print(f"Loaded {len(val_dataset)} validation images.")
    print(f"Loaded {len(test_dataset)} test images.")
    print(f"Detected classes: {classes}")
    
    # Calculate Class Weights for Weighted CrossEntropyLoss
    class_weights = get_class_weights(train_dataset).to(device)
    print(f"Computed Class Weights: {class_weights.cpu().numpy()}")
    
    # 4. Instantiate Model
    model = CustomResidualCNN(num_classes=len(classes)).to(device)
    
    # 5. Define Loss, Optimizer, Scheduler, EarlyStopping, Scaler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # ReduceLROnPlateau reduces learning rate when validation loss stops improving
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    
    # EarlyStopping prevents memorization in late epochs
    early_stopping = EarlyStopping(patience=3, verbose=True, path="residual_model.pth")
    
    # AMP Scaler for CUDA (MPS doesn't use scaler)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # Track metrics
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }
    
    print("\n" + "="*50)
    print("           TRAINING CUSTOMRESIDUALCNN (MODEL 5)")
    print("="*50)
    
    # 6. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            elif device.type == 'mps':
                try:
                    with torch.amp.autocast('mps'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                except Exception:
                    # Fallback if mps autocast not supported
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        
        # Validation evaluation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                    
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_running_loss / len(val_dataset)
        val_acc = val_correct / val_total
        
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(epoch_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
              
        # Scheduler step based on validation loss
        scheduler.step(val_loss)
        
        # Early Stopping Check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Model convergence stalled.")
            break
            
    print("\nTraining completed! Best Validation Loss:", early_stopping.val_loss_min)
    
    # 7. Plot and Save Learning Curves
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    actual_epochs = len(history["train_loss"])
    plt.plot(range(1, actual_epochs + 1), history["train_loss"], label="Train Loss", color='blue')
    plt.plot(range(1, actual_epochs + 1), history["val_loss"], label="Val Loss", color='red')
    plt.title("Loss Curves (CustomResidualCNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, actual_epochs + 1), history["train_acc"], label="Train Accuracy", color='blue')
    plt.plot(range(1, actual_epochs + 1), history["val_acc"], label="Val Accuracy", color='red')
    plt.title("Accuracy Curves (CustomResidualCNN)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("residual_curves.png")
    print("Saved learning curves as 'residual_curves.png'.")
    
    # 8. Load Best Checkpoint and Evaluate on Test Set
    print("\n" + "="*50)
    print("           EVALUATION ON UNSEEN TEST SET")
    print("="*50)
    
    model.load_state_dict(torch.load("residual_model.pth"))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Print Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Print Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
