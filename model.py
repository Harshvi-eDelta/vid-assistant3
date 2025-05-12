'''import torch
import torch.nn as nn
import torch.nn.functional as F

class LandmarkCNN(nn.Module):
    def __init__(self, num_landmarks=68):
        super(LandmarkCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_landmarks * 2)  # 68 landmarks × 2 (x, y)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128x128
        x = self.pool(F.relu(self.conv2(x)))  # 64x64
        x = self.pool(F.relu(self.conv3(x)))  # 32x32
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x'''

# heatmap - multiple stage
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapRefineBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 68, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

class MultiStageCNN(nn.Module):
    def __init__(self, stages=5):
        super().__init__()
        self.stages = stages
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64
        )
        self.refine_blocks = nn.ModuleList([
            HeatmapRefineBlock(in_channels=128 if i == 0 else 128 + 68) for i in range(stages)
        ])

    # def forward(self, x):
    #     feat = self.stem(x)
    #     heatmaps = torch.zeros(x.size(0), 68, 64, 64).to(x.device)
    #     outputs = []

    #     for i in range(self.stages):
    #         inp = feat if i == 0 else torch.cat([feat, heatmaps], dim=1)
    #         heatmaps = self.refine_blocks[i](inp)
    #         outputs.append(heatmaps)

    #     return outputs  # List of 5 stages

    def forward(self, x):
        feat = self.stem(x)  # 128x64x64
        heatmaps = torch.zeros(x.size(0), 68, 64, 64).to(x.device)
        outputs = []

        for i in range(self.stages):
            inp = feat if i == 0 else torch.cat([feat, heatmaps], dim=1)  # All 64x64
            heatmaps = self.refine_blocks[i](inp)
            outputs.append(F.interpolate(heatmaps, size=(128, 128), mode='bilinear', align_corners=False))

        return outputs  # All are 68×128×128


