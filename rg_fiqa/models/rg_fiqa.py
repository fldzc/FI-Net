import torch
import torch.nn as nn
import torch.nn.functional as F

class RGFIQA(nn.Module):
    """
    RG-FIQA: Rule-Guided Face Image Quality Assessment.
    A lightweight Face Quality network trained via rule-guided knowledge distillation.
    Parameters: ~1.3M
    Input: (3, 112, 112) RGB
    Output: A single quality score [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)      # 112×112
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)     # 56×56
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)    # 28×28
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)   # 14×14
        self.bn4   = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)   # 7×7
        self.bn5   = nn.BatchNorm2d(512)

        self.avg   = nn.AdaptiveAvgPool2d(1)        # 1×1
        self.fc    = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = self.avg(x).view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))               # 0–1
        return x
