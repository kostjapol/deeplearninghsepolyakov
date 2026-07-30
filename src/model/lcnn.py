import torch
from torch import nn


class MFM(nn.Module):
    def forward(self, x):
        first_half, second_half = torch.chunk(x, 2, dim=1)
        return torch.maximum(first_half, second_half)


class ConvMFM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=padding,
        )

    def forward(self, x):
        x = self.conv(x)
        return MFM()(x)


class LCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            ConvMFM(1, 32, 5, padding=2),
            nn.MaxPool2d(2),
            ConvMFM(32, 32, 1),
            nn.BatchNorm2d(32),
            ConvMFM(32, 48, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(48),
            ConvMFM(48, 48, 1),
            nn.BatchNorm2d(48),
            ConvMFM(48, 64, 3, padding=1),
            nn.MaxPool2d(2),
            ConvMFM(64, 64, 1),
            nn.BatchNorm2d(64),
            ConvMFM(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            ConvMFM(32, 32, 1),
            nn.BatchNorm2d(32),
            ConvMFM(32, 32, 3, padding=1),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 53 * 37, 160),
            MFM(),
            nn.Dropout(0.75),
            nn.BatchNorm1d(80),
            nn.Linear(80, 2),
        )

        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, data_object, **batch):
        x = self.features(data_object)
        logits = self.classifier(x)

        return {"logits": logits}
