import os
import torch
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class TargetedAugmentationDataset(datasets.ImageFolder):
    """
    Applies standard transforms to most classes, but extremely heavy augmentation
    specifically to 'ivy guard' to prevent overfitting on the minority class.
    """
    def __init__(self, root, standard_transform=None, ivy_transform=None):
        super().__init__(root)
        self.standard_transform = standard_transform
        self.ivy_transform = ivy_transform
        
        # Find index for 'ivy guard'
        self.ivy_idx = None
        for idx, class_name in enumerate(self.classes):
            if class_name.lower() in ['ivy guard', 'ivy gourd']:
                self.ivy_idx = idx
                break

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.ivy_idx is not None and target == self.ivy_idx and self.ivy_transform is not None:
            sample = self.ivy_transform(sample)
        elif self.standard_transform is not None:
            sample = self.standard_transform(sample)
            
        return sample, target

def get_class_weights(dataset):
    """
    Calculates inverse class frequencies to use in CrossEntropyLoss.
    """
    class_counts = np.zeros(len(dataset.classes))
    for _, target in dataset.samples:
        class_counts[target] += 1
        
    # Prevent division by zero
    class_counts = np.where(class_counts == 0, 1, class_counts)
    
    # Calculate inverse frequencies
    weights = 1.0 / class_counts
    
    # Normalize weights so they sum to num_classes
    weights = weights * len(dataset.classes) / np.sum(weights)
    
    return torch.FloatTensor(weights)
