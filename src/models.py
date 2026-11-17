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

class WildfireCNN_iteration1(nn.Module):
    """
    Iteration 1 2D-CNN: Preserves spatial resolution with padded convolutions and
    fuses center target-cell features with global 35km x 35km neighborhood context.
    Input: (Batch, 9, 7, 7) -> Output: (Batch, 1) raw logits.
    """
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        # Block 1: (Batch, 9, 7, 7) -> (Batch, 32, 7, 7)
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = nn.GELU()

        # Block 2: (Batch, 32, 7, 7) -> (Batch, 64, 7, 7)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = nn.GELU()

        # Block 3 (Refinement with Residual Skip): (Batch, 64, 7, 7) -> (Batch, 64, 7, 7)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.GELU()

        # Global spatial pooling for neighborhood summary: (Batch, 64, 7, 7) -> (Batch, 64, 1, 1)
        self.context_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head: fuses 64 center-cell features + 64 neighborhood features = 128
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(p=dropout_p * 0.5),
            nn.Linear(32, 1)  # Raw logit output
        )

    def forward(self, x):
        # Spatial feature extraction preserving 7x7 grid alignment
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        
        # Residual block around conv3
        residual = x
        x = self.act3(self.bn3(self.conv3(x)) + residual)

        # 1. Center target-cell representation at index (3, 3) -> (Batch, 64)
        center_feat = x[:, :, 3, 3]

        # 2. Global 35km x 35km neighborhood context -> (Batch, 64)
        context_feat = self.context_pool(x).flatten(start_dim=1)

        # Concatenate local center pixel + regional spatial context -> (Batch, 128)
        fused = torch.cat([center_feat, context_feat], dim=1)
        return self.fc(fused)