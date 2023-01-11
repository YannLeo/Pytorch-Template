import torch
from torch import nn
import torchvision


class MNISTEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.feature_extractor = torchvision.models.resnet18(pretrained=False, num_classes=out_dim)
    
    def forward(self, x):
        return self.feature_extractor(x)
