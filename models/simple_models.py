from typing import Literal, Optional
from torch import nn
import torch


def Projector(in_dim, out_dim, intermediate_dim=128, layers: Literal[2, 3] = 2):
    __inter_layers = [nn.Linear(intermediate_dim, intermediate_dim), nn.LeakyReLU(0.03)] if layers == 3 else []
    return nn.Sequential(
        nn.Linear(in_dim, intermediate_dim), nn.LeakyReLU(0.03),
        *__inter_layers,
        nn.Linear(intermediate_dim, out_dim)
    )


def Classifier(in_dim, num_class, intermediate_dim=128, layers: Literal[2, 3] = 2):
    __inter_layers = [nn.Linear(intermediate_dim, intermediate_dim), nn.LeakyReLU(0.03)] if layers == 3 else []
    return nn.Sequential(
        nn.Linear(in_dim, intermediate_dim),
        nn.LeakyReLU(0.03),
        *__inter_layers,
        # nn.Dropout(0.5),
        nn.Linear(intermediate_dim, num_class)
    )


class VIB(nn.Module):
    """Variational Information Bottleneck (VIB)"""

    def __init__(self, in_dim, out_dim, lambda_=0.05,
                 pre_activation: Optional[nn.Module] = None) -> None:
        """
        Args:
          in_dim: the dimension of the input
          out_dim: the dimension of the output
          lambda_: This is the weight decay parameter.
          pre_activation: This is the activation function that is applied to the input before the mean
        and variance are computed, e.g., nn.LeakyReLU(0.03).
        """
        super().__init__()
        self.mu_net = nn.Linear(in_dim, out_dim)
        self.log_var_net = nn.Linear(in_dim, out_dim)
        self.lambda_ = lambda_
        self.pre_activation = pre_activation  # e.g., nn.LeakyReLU(0.03)

    def forward(self, x, training=True):
        if self.pre_activation is not None:
            x = self.pre_activation(x)

        mu = self.mu_net(x)
        if not training:
            return mu

        log_var = self.log_var_net(x)
        kl_loss = -0.5 * self.lambda_ * \
            torch.mean(torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var), dim=1))
        return self.reparameterize(mu, log_var), kl_loss

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean


def SimpleCNN1D():
    return nn.Sequential(
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
        nn.Flatten(),  # (,256)
        nn.ELU(),
        nn.Linear(256, 128)
    )


if __name__ == '__main__':
    cnn = SimpleCNN1D()
    x = torch.rand(3, 2, 256)
    print(cnn(x).shape)
