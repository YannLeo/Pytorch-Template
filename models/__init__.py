from .resnet1D import ResNet1D
from .resnext1D import ResNext1D
from .WiSig_nets import WiSigNet_MINE, WiSigResNet_MINE, WiSigEncoder
from .mnist_nets import MNISTResNet_MINE, MNISTEncoder
from .simple_models import Classifier, Projector, VIB, SimpleCNN1D, SimpleCNN2D


__all__ = [
    'ResNet1D',
    'ResNext1D',
    'WiSigNet_MINE', 'WiSigResNet_MINE', 'WiSigEncoder'
    'MNISTResNet_MINE', 'MNISTEncoder',
    'Classifier', 'Projector', 'VIB', 'SimpleCNN1D', 'SimpleCNN2D'
]
