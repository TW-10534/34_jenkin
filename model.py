# model.py
import torch.nn as nn
import torch.nn.functional as F

class SimpleCIFARConvNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Input: [B, 3, 32, 32]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # halves H, W

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # After 2x pooling: 32x32 -> 16x16 -> 8x8 (if we pool twice)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # [B, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 64, 8, 8]
        x = F.relu(self.conv3(x))             # -> [B, 128, 8, 8]

        x = x.view(x.size(0), -1)             # -> [B, 128*8*8]
        x = F.relu(self.fc1(x))               # -> [B, 256]
        x = self.fc2(x)                       # -> [B, num_classes]
        return x
