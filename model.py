import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Squeeze-and-Excitation block (channel-wise recalibration)
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64 // 16, 1),  # bottleneck
            nn.ReLU(),
            nn.Conv2d(64 // 16, 64, 1),
            nn.Sigmoid()
        )

        # Dynamically compute flatten dimension
        self.flatten_dim = self._get_flatten_dim()

        self.fc_block = nn.Sequential(
            nn.Linear(self.flatten_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

    def _get_flatten_dim(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 128, 256)
            x = self.conv_block1(x)
            x = self.conv_block2(x)
            x = self.conv_block3(x)

            se_weight = self.se_block(x)
            x = x * se_weight

            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        se_weight = self.se_block(x)
        x = x * se_weight

        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x
