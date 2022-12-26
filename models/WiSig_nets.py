import torch
from torch import nn
from .resnet1D import ResNet1D


class WiSigNet_MINE(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            nn.Conv1d(2, 8, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 16, 3, 1, 1),
            nn.Flatten()
        )
        self.projector = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 32))
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.bn = torch.nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.feat_extractor(x)
        feat = self.projector(x)
        if self.bn:
            feat = self.bn(feat)
        out = self.linear(feat)
        return out, feat


class WiSigResNet_MINE(WiSigNet_MINE):
    # Change the feature_extractor model from normal CNN1D to resnet1D18
    def __init__(self, num_classes=6) -> None:
        super().__init__(num_classes)
        self.feature_extractor = ResNet1D(
            in_channels=2,
            base_filters=64,
            kernel_size=3,
            stride=2,
            n_block=4,
            groups=1,
            n_classes=256,
            downsample_gap=2,
            increasefilter_gap=1,
            verbose=False
        )


class WiSigEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.feature_extractor = ResNet1D(
            in_channels=2,
            base_filters=64,
            kernel_size=3,
            stride=2,
            n_block=4,
            groups=1,
            n_classes=out_dim,
            downsample_gap=2,
            increasefilter_gap=1,
            verbose=False
        )
    
    def forward(self, x):
        return self.feature_extractor(x)
        