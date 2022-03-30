"""Models."""

import torch
from torch import Tensor


class HighBetaMLP(torch.nn.Module):
    def __init__(self, width: int = 16, a: float = 1.0, psi_0: float = 1.0) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(2, width)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(width, 1)

        #  Initialize last bias to unit, since psi(r=0)=psi_0
        torch.nn.init.ones_(self.fc2.bias)
        torch.nn.init.normal_(self.fc2.weight, std=3e-2)

        #  Scaling parameters
        self.a = a
        self.psi_0 = psi_0

    def forward(self, x):
        rho = x[:, 0] / self.a
        theta = x[:, 1]
        #  Compute features
        x1 = rho * torch.cos(theta)
        x2 = rho * torch.sin(theta)
        #  Compute psi
        psi_hat = self.fc1(torch.stack((x1, x2), dim=1))
        psi_hat = self.tanh(psi_hat / 2)
        return self.psi_0 * self.fc2(psi_hat).view(-1)


class GradShafranovMLP(torch.nn.Module):
    def __init__(
        self,
        width: int = 32,
        R0: float = 0.0,
        a: float = 1.0,
        b: float = 1.0,
        psi_0: float = 1.0,
    ) -> None:
        super().__init__()

        self.fc1 = torch.nn.Linear(2, width)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(width, 1)

        #  Scaling parameters
        self.R0 = R0
        self.a = a
        self.b = b
        self.psi_0 = psi_0

    def forward(self, x: Tensor) -> Tensor:
        R = x[:, 0]
        Z = x[:, 1]
        #  Scale features
        R = (R - self.R0) / self.a
        Z = Z / self.b
        #  Compute psi
        psi_hat = self.fc1(torch.stack([R, Z], dim=-1))
        psi_hat = self.tanh(psi_hat / 2)
        return self.psi_0 * self.fc2(psi_hat).view(-1)
