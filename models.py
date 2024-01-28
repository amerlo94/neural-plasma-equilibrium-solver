"""Models."""

from typing import Union

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
        R0: float = 1.0,
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

        #  Initialize last bias to zero, since psi(Ra, Za)=0
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.normal_(self.fc2.weight, std=3e-2)

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

    def find_x_of_psi(
        self,
        psi: Union[float, str],
        initial_guess: Tensor,
        tolerance: float = 1e-5,
        tolerance_change: float = 1e-8,
    ):
        """
        Find domain value x_0 such that psi = self(x_0).

        If psi is "min" or "max", find domain value where minimize or maximize self.
        """

        assert initial_guess.shape == (1, 2)

        if isinstance(psi, str):
            assert psi in ("min", "max")
        else:
            psi = torch.as_tensor(psi)

        #  Copy tensor to avoid changes to the original one
        initial_guess = torch.Tensor(initial_guess)
        initial_guess.requires_grad_(True)

        #  Disable model graph
        self.requires_grad_(False)

        #  Create optimization loop to finx x
        optim = torch.optim.LBFGS([initial_guess], lr=1e-2)

        def closure():
            optim.zero_grad()
            psi_hat = self.forward(initial_guess)
            if psi == "min":
                loss = psi_hat
            elif psi == "max":
                loss = -psi_hat
            else:
                loss = torch.abs(psi_hat - psi)
            loss.backward()
            return loss

        #  Compute first iteration
        psi_hat = self.forward(initial_guess)

        while True:

            #  Update initial guess
            psi_hat_old = psi_hat
            optim.step(closure)

            #  Check for convergence
            psi_hat = self.forward(initial_guess)
            if not isinstance(psi, str):
                if torch.abs(psi_hat - psi) <= tolerance:
                    break
            if torch.abs(psi_hat - psi_hat_old) <= tolerance_change:
                break

        #  Enable back model graph
        self.requires_grad_(True)

        return initial_guess.detach()


class InverseGradShafranovMLP(torch.nn.Module):
    def __init__(
        self,
        Rb,
        Zb,
        width: int = 16,
        num_features: int = 12,
    ) -> None:
        super().__init__()

        #  Fourier features
        self.num_features = num_features
        self.modes = torch.arange(num_features).view(-1, num_features)

        self.R_branch = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, num_features, bias=False),
        )
        self.l_branch = torch.nn.Sequential(
            torch.nn.Linear(1, num_features),
        )
        self.Z_branch = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, num_features, bias=False),
        )
        pad = torch.zeros(num_features)
        self.Rb = pad.clone()
        self.Rb[: len(Rb)] = Rb
        self.Rb = self.Rb.view(-1, num_features)
        self.Zb = pad.clone()
        self.Zb[: len(Zb)] = Zb
        self.Zb = self.Zb.view(-1, num_features)
        self.lb = torch.ones(num_features)
        self.lb[0] = 0
        self.lb = self.lb.view(-1, num_features)

        #  Initialize layers
        for tensor in (
            self.R_branch[-1].weight,
            self.Z_branch[-1].weight,
            self.l_branch[-1].weight,
        ):
            torch.nn.init.normal_(tensor=tensor, std=1e-2)
        for tensor in (self.l_branch[-1].bias,):
            torch.nn.init.constant_(tensor=tensor, val=1e-2)

    def forward(self, x: Tensor) -> Tensor:
        rho = x[:, 0].view(-1, 1)
        theta = x[:, 1].view(-1, 1)
        #  Get Fourier Features
        rf = theta * self.modes
        cosm = torch.cos(rf)
        sinm = torch.sin(rf)
        #  Compute R, lambda and Z
        rho_factor = rho**self.modes
        normalized_rho = 2 * rho**2 - 1
        R = self.Rb * rho_factor * (1 + (1 - rho**2) * self.R_branch(normalized_rho))
        R = (R * cosm).sum(dim=1).view(-1, 1)
        l = self.lb * rho_factor * (self.l_branch(normalized_rho))
        l = (l * sinm).sum(dim=1).view(-1, 1)
        Z = self.Zb * rho_factor * (1 + (1 - rho**2) * self.Z_branch(normalized_rho))
        Z = (Z * sinm).sum(dim=1).view(-1, 1)
        #  Build model output
        RlZ = torch.cat([R, l, Z], dim=-1)
        return RlZ
