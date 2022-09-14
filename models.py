"""Models."""

from typing import Union, Tuple

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
        num_features: int = 3,
    ) -> None:
        super().__init__()

        #  Fourier features
        self.num_features = num_features
        self.B = torch.arange(num_features).view(-1, num_features)

        self.R_branch = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, num_features),
        )
        self.l_branch = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, num_features),
        )
        self.Z_branch = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, num_features),
        )

        #  Boundary condition used as scaling factors
        #  Use default small values in case coefficient is not defined in boundary
        pad = torch.ones(num_features) * 1e-3
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
            torch.nn.init.normal_(tensor, std=1e-2)
        for tensor in (
            self.R_branch[-1].bias,
            self.l_branch[-1].bias,
            self.Z_branch[-1].bias,
        ):
            torch.nn.init.zeros_(tensor)

    def forward(self, x: Tensor) -> Tensor:
        rho = x[:, 0].view(-1, 1)
        theta = x[:, 1].view(-1, 1)
        #  Get random Fourier Features
        rf = theta * self.B
        cosm = torch.cos(rf)
        sinm = torch.sin(rf)
        #  Compute R, lambda and Z
        rho_factor = torch.cat([rho**m for m in range(self.num_features)], dim=-1)
        R = self.Rb * rho_factor * (1 + self.R_branch(rho))
        R = (R * cosm).sum(dim=1).view(-1, 1)
        l = self.lb * rho_factor * (1 + self.l_branch(rho))
        l = (l * sinm).sum(dim=1).view(-1, 1)
        Z = self.Zb * rho_factor * (1 + self.Z_branch(rho))
        Z = (Z * sinm).sum(dim=1).view(-1, 1)
        #  Build model output
        RlZ = torch.cat([R, l, Z], dim=-1)
        return RlZ


#  TODO: rename it Legendre model
class Inverse3DMHDMLP(torch.nn.Module):
    def __init__(
        self,
        Rb,
        Zb,
        Ra,
        Za,
        nfp: int,
        sym: bool = True,
        #  TODO: add argument here in train and set to 1 by default
        lrad: int = 2,
        mpol: int = 1,
        ntor: int = 1,
    ) -> None:
        super().__init__()

        assert sym, NotImplementedError("non-symmetric not implemented yet")

        self.lrad = lrad
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp

        #  Build Fourier modes
        poloidal_modes = torch.arange(self.mpol + 1)
        toroidal_modes = torch.arange(-self.ntor, self.ntor + 1) * nfp
        mn = torch.cartesian_prod(poloidal_modes, toroidal_modes)[self.ntor :]
        self.xm = mn[:, 0]
        self.xn = mn[:, 1]

        #  Legendre polynomial coefficients
        #  For R and Z, the last Legendre coefficients if fixed so to satisfy the
        #  boundary condition
        self.rmnl = torch.nn.parameter.Parameter(torch.randn(len(mn), self.lrad))
        self.lmnl = torch.nn.parameter.Parameter(torch.randn(len(mn), self.lrad + 1))
        self.zmnl = torch.nn.parameter.Parameter(torch.randn(len(mn), self.lrad))

        #  TODO: they need to be normalized
        def legendre_fn(s: Tensor) -> Tensor:
            s = 2 * s - 1
            s = s.view(-1)
            polys = [torch.ones_like(s), s]
            for n in range(1, self.lrad):
                poly = ((2 * n + 1) * s * polys[n] - n * polys[n - 1]) / (n + 1)
                polys.append(poly)
            return torch.stack(polys, dim=-1)

        self.legendre = legendre_fn

        self.Rb = Rb
        self.Zb = Zb

        self.Ra = torch.nn.functional.pad(Ra, (0, len(mn) - len(Ra)), value=0)
        self.Za = torch.nn.functional.pad(Za, (0, len(mn) - len(Ra)), value=0)

        #  Save Fourier coefficients to compute spectral condensation
        self._rmnc = None
        self._zmns = None

        #  Initialize model
        for tensor in (
            self.rmnl,
            self.lmnl,
            self.zmnl,
        ):
            torch.nn.init.zeros_(tensor)

        with torch.no_grad():
            #  m=0 modes:
            #  linear interpolation between the magnetic axis and the boundary
            self.rmnl.data[: self.ntor + 1, 0] = 0.5 * (
                self.Rb[: self.ntor + 1] + self.Ra[: self.ntor + 1]
            )
            self.zmnl.data[: self.ntor + 1, 0] = 0.5 * (
                self.Zb[: self.ntor + 1] + self.Za[: self.ntor + 1]
            )
            #  satisfy axis guess at initialization
            if self.lrad >= 2:
                self.rmnl.data[: self.ntor + 1, 1] = 0.5 * (
                    self.Rb[: self.ntor + 1] - self.Ra[: self.ntor + 1]
                )
                self.zmnl.data[: self.ntor + 1, 1] = 0.5 * (
                    self.Zb[: self.ntor + 1] - self.Za[: self.ntor + 1]
                )
            #  m>0 modes:
            #  rho**m Xb as initial guess
            self.rmnl.data[self.ntor + 1 :, 0] = self.Rb[self.ntor + 1 :]
            self.zmnl.data[self.ntor + 1 :, 0] = self.Zb[self.ntor + 1 :]

        ############################
        # TODO: remove me! Only used for debugging during training
        # import matplotlib.pyplot as plt
        # import numpy as np

        # plt.ion()
        # # self._fig, ax = plt.subplots(1, 1, tight_layout=True)
        # # self._idx = 0
        # # (self._line1,) = ax.plot(
        # #     np.linspace(0, 1, 99),
        # #     np.linspace(self.Ra[self._idx], self.Rb[self._idx], 99),
        # # )
        # self._fig, self._ax = plt.subplots(1, 1, tight_layout=True)
        # self._t = [0]
        # self._data = [self.rmnl[0, 0].detach()]
        # (self._line1,) = self._ax.plot(self._t, self._data)
        ############################

    def forward(self, x: Tensor) -> Tensor:

        rho = x[:, 0]
        theta = x[:, 1]
        zeta = x[:, 2]

        angle = theta[:, None] * self.xm[None, :] - zeta[:, None] * self.xn[None, :]
        costzmn = torch.cos(angle)
        sintzmn = torch.sin(angle)

        rho_factor = torch.stack([rho**m for m in self.xm], dim=-1)

        bases = self.legendre(rho**2).view(-1, 1, self.lrad + 1)

        #  Satisfy boundary
        #  When a function is expanded in Legendre polynomials,
        #  the value at s=1 is simply the sum of the polynomial coefficients.
        #  f(s=1) = x0 + x1 + x2 + x3 + ...
        #  Set the highest order term to satisfy the boundary by construction
        #  xn = Xb - (x0 + x1 + x2 + x3 + ...)
        rmnl = torch.cat(
            [self.rmnl, (self.Rb - self.rmnl.sum(dim=-1))[..., None]],
            dim=-1,
        )
        zmnl = torch.cat(
            [self.zmnl, (self.Zb - self.zmnl.sum(dim=-1))[..., None]],
            dim=-1,
        )

        rmnc = rho_factor * (bases * rmnl[None, ...]).sum(dim=-1)
        R = (costzmn * rmnc).sum(dim=-1)

        lmns = rho_factor * (bases * self.lmnl[None, ...]).sum(dim=-1)
        #  Lamda is a periodic function
        lmns[:, 0] = 0
        l = (sintzmn * lmns).sum(dim=-1)

        zmns = rho_factor * (bases * zmnl[None, ...]).sum(dim=-1)
        #  Assume stellarator symmetry
        zmns[:, 0] = 0
        Z = (sintzmn * zmns).sum(dim=-1)

        #######################
        # TODO: remove me!
        # self._t.append(self._t[-1] + 1)
        # self._data.append(self.rmnl[0, 0].detach().item())
        # self._line1.set_xdata(self._t)
        # self._line1.set_ydata(self._data)
        # # self._line1.set_xdata(rho.detach() ** 2)
        # # self._line1.set_ydata(rmnc[:, self._idx].detach())
        # self._fig.canvas.draw()
        # self._fig.canvas.flush_events()
        # self._ax.relim()
        # self._ax.autoscale_view()
        #######################

        #  Save Fourier modes
        self._rmnc = rmnc
        self._zmns = zmns

        RlZ = torch.stack([R, l, Z], dim=-1)
        return RlZ

    def M(self, p: int = 2, q: int = 0) -> Tensor:
        """
        Spectral condensation measure.

        TODO: consider to add me as utils.
        """
        energy = self._rmnc**2 + self._zmns**2
        num = self.xm ** (p + q) * energy
        if q == 0:
            print(num.sum())
            return num.sum()
        denom = self.xm**p * energy
        return num.sum() / denom.sum()
