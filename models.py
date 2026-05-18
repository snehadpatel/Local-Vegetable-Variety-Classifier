import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# Model 1 — Basic CNN (your baseline)
# =====================================================================
class CustomLightCNN(nn.Module):
    """
    3 conv blocks, GAP, and FC layer. Baseline floor model.
    """
    def __init__(self, num_classes=5):
        super(CustomLightCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.6)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# =====================================================================
# Model 2 — Deep CNN with Batch Normalisation
# =====================================================================
class CustomDeepCNN(nn.Module):
    """
    5 conv blocks, BatchNorm after each, Dropout before final FC layer.
    """
    def __init__(self, num_classes=5):
        super(CustomDeepCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # FC head: 256 * 7 * 7 = 12544
        self.fc1 = nn.Linear(256 * 7 * 7, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =====================================================================
# Model 3 — CNN + Global Average Pooling
# =====================================================================
class CustomGAPCNN(nn.Module):
    """
    Same 5 conv blocks, but Global Average Pooling cuts millions of parameters.
    """
    def __init__(self, num_classes=5):
        super(CustomGAPCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global Average Pooling collapses 7x7 spatial layout -> 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)  # 256 inputs
        x = self.fc(x)
        return x


# =====================================================================
# Model 4 — Multi-Scale CNN (Inception-inspired)
# =====================================================================
class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Three parallel paths
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(out_ch)
        
        self.conv5 = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2)
        self.bn5   = nn.BatchNorm2d(out_ch)
        
        self.conv7 = nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3)
        self.bn7   = nn.BatchNorm2d(out_ch)
        
        self.dropout = nn.Dropout2d(p=0.2)
        self.relu  = nn.ReLU()

    def forward(self, x):
        b3 = self.bn3(self.conv3(x))
        b5 = self.bn5(self.conv5(x))
        b7 = self.bn7(self.conv7(x))
        out = torch.cat([b3, b5, b7], dim=1)  # concat on channel axis
        out = self.dropout(out)
        return self.relu(out)


class CustomMultiScaleCNN(nn.Module):
    """
    Inception-inspired multi-scale convolutional network.
    """
    def __init__(self, num_classes=5):
        super(CustomMultiScaleCNN, self).__init__()
        
        # Block 1: 3 -> 16*3 = 48 channels
        self.ms1 = MultiScaleBlock(3, 16)
        # Block 2: 48 -> 32*3 = 96 channels
        self.ms2 = MultiScaleBlock(48, 32)
        # Block 3: 96 -> 64*3 = 192 channels
        self.ms3 = MultiScaleBlock(96, 64)
        # Block 4: 192 -> 128*3 = 384 channels
        self.ms4 = MultiScaleBlock(192, 128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(384, num_classes)
        
    def forward(self, x):
        x = self.pool(self.ms1(x))
        x = self.pool(self.ms2(x))
        x = self.pool(self.ms3(x))
        x = self.pool(self.ms4(x))
        
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


# =====================================================================
# Model 5 — CNN with your own Skip Connections
# =====================================================================
class CustomResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CustomResidualCNN(nn.Module):
    """
    Scratch ResNet model with bypass links.
    """
    def __init__(self, num_classes=5):
        super(CustomResidualCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.res1 = CustomResidualBlock(32, 32, stride=1)
        self.res2 = CustomResidualBlock(32, 64, stride=1)
        self.res3 = CustomResidualBlock(64, 128, stride=1)
        self.res4 = CustomResidualBlock(128, 256, stride=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(256, num_classes)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.res1(x))
        x = self.pool(self.res2(x))
        x = self.pool(self.res3(x))
        x = self.pool(self.res4(x))
        
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# =====================================================================
# Model 6 — Depthwise Separable CNN
# =====================================================================
class DepthwiseSepConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise  = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise  = nn.Conv2d(in_ch, out_ch, 1)
        self.bn         = nn.BatchNorm2d(out_ch)
        self.relu       = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)   # filter each channel independently
        x = self.pointwise(x)   # mix channels with 1×1 conv
        return self.relu(self.bn(x))


class CustomDepthwiseSepCNN(nn.Module):
    """
    Lightweight, fast Depthwise Separable CNN (MobileNet-style concept).
    """
    def __init__(self, num_classes=5):
        super(CustomDepthwiseSepCNN, self).__init__()
        
        self.ds1 = DepthwiseSepConv(3, 16)
        self.ds2 = DepthwiseSepConv(16, 32)
        self.ds3 = DepthwiseSepConv(32, 64)
        self.ds4 = DepthwiseSepConv(64, 128)
        self.ds5 = DepthwiseSepConv(128, 256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(self.ds1(x))
        x = self.pool(self.ds2(x))
        x = self.pool(self.ds3(x))
        x = self.pool(self.ds4(x))
        x = self.pool(self.ds5(x))
        
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
