import torch
from torch import nn
import torchvision


class MNISTResNet_MINE(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.feat_extractor = torchvision.models.resnet18(pretrained=False, num_classes=256)
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

class MNISTEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.feature_extractor = torchvision.models.resnet18(pretrained=False, num_classes=out_dim)
    
    def forward(self, x):
        return self.feature_extractor(x)
