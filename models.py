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
        width: int = 16,
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
    #  (R,Z) = NN(s, theta) with s some radial flux coordinate
    #    here chosen as r=sqrt(psi/psi_0) - same as DESC
    def __init__(self,
                 width: int = 16,
                 R0: float = 0.0,
                 a = 0., b = 0.
                 ) -> None:
        super().__init__()

        self.R0 = R0
        self.a, self.b = a, b

        self.fc0 = torch.nn.Linear(2, width)
        self.relu = torch.nn.Tanh()
        self.fc1 = torch.nn.Linear(width, width)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(width, 3)

        #  Initialize last bias to zero, since NN(0,0)=0
        torch.nn.init.zeros_(self.fc2.bias)
        # with torch.no_grad():
        #     self.fc2.bias[0].data = self.R0
        torch.nn.init.normal_(self.fc2.weight, std=2e-3)

        self.forward = self.forward_

    def forward_(self, x: Tensor) -> Tensor:
        # x = torch.cat((
        #     x[:, 0].unsqueeze(1),
        #     torch.sin(x[:, 1]).unsqueeze(1),
        #     torch.cos(x[:, 1]).unsqueeze(1)
        # ), dim=1)

        # x = torch.cat((x[:,0].unsqueeze(1),
        #    (x[:,1]/(torch.pi)).unsqueeze(1)), dim=1)

        out = self.fc0(x)
        out = self.relu(out / 2)
        out = self.fc1(out)
        out = self.tanh(out / 2)
        out = self.fc2(out)  # * x[:, :1]
        out[:, 0] = out[:, 0] + self.R0
        return out

    def forward_sym(self, x: Tensor) -> Tensor:
        # s = x[:, 0]
        # theta = x[:, 1]

        # x = torch.cat((
        #     x[:, 0].unsqueeze(1),
        #     torch.sin(x[:, 1]).unsqueeze(1),
        #     torch.cos(x[:, 1]).unsqueeze(1)
        # ), dim=1)

        # x = torch.cat((x[:,0].unsqueeze(1),
        #    (x[:,1]/(torch.pi)).unsqueeze(1)), dim=1)

        #out = x[x[:, 1] % torch.pi != 0]  # only theta > 0 components, upper half plane (Z>0)
        out = x[x[:, 1] >= 0].clone()
        idc_z0 = (out[:, 1] % torch.pi == 0).nonzero(as_tuple=True)[0]
        z0_mask = torch.ones_like(out[:,0], dtype=torch.bool)
        z0_mask[idc_z0] = 0
        #out = torch.vstack((out, x[x[:, 1] % torch.pi == 0]))  # points on R,Z=0 axis
        out = self.fc0(out)
        out = self.relu(out/2)
        out = self.fc1(out)
        out = self.tanh(out/2)
        out = self.fc2(out)  # * x[:, :1]

        out[:, 0] = out[:, 0] + self.R0

        # predict 1/(2*Nfp) R,Z,lambda (up-down symmetry) - then mirror
        idc_neg = (x[:, 1] < 0).nonzero(as_tuple=True)[0]
        idc_pos = (x[:, 1] >= 0).nonzero(as_tuple=True)[0]
        out_mirror = torch.zeros(size=(x.shape[0], 3))
        out_mirror[idc_pos] = out.clone()
        out_mirror[idc_neg] = out[z0_mask].clone()
        out_mirror[idc_neg, 1] *= -1  # Z = -Z in stellarator symmetry when flipping theta
        # out_mirror[idc_0] = out[-len(idc_0):]
        # out_mirror[idc_neg] = out[:-len(idc_0)]
        # out_mirror[idc_neg, 1] *= -1
        # out_mirror[idc_neg] = out

        return out_mirror




