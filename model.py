from torchvision.models import resnet50
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, num_class=8):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, num_class))
        
    def forward(self, x):
        out = self.resnet(x)
        return out


if __name__ == '__main__':
    net = ResNet50()
    print(net)