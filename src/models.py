import torch
import torch.nn as nn

class WildfireCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # In: (Batch, 9, 7, 7) -> Out: (Batch, 32, 5, 5)
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        # In: (Batch, 32, 5, 5) -> Out: (Batch, 64, 3, 3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        # Spatial pooling: collapses (3, 3) -> (1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 1)  # Raw logit output
        )

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return self.fc(x)

from torchvision.models import resnet50


class WildfireResNet50(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize ResNet-50 without pretrained weights
        self.backbone = resnet50(weights=None)

        # Adapt input for nine environmental channels
        self.backbone.conv1 = nn.Conv2d(
            in_channels=9,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Preserve spatial information in small patches
        self.backbone.maxpool = nn.Identity()

        # Binary classification head
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            1
        )

    def forward(self, x):
        return self.backbone(x)