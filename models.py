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


class Inverse3DMHDMLP(torch.nn.Module):
    def __init__(
        self,
        Rb,
        Zb,
        ns: int,
        ntheta: int,
        nzeta: int,
        nfp: int,
        sym: bool = True,
        width: int = 16,
        # if m=2 then m ∈ {0, 1, 2}
        max_mpol: int = 1,
        # if n=1 then n ∈ {-1, 0, 1}, max_ntor is highest positive toroidal mode number in output
        max_ntor: int = 1,
    ) -> None:
        super().__init__()

        assert sym, NotImplementedError("non-symmetric not implemented yet")

        self.max_mpol = max_mpol
        self.max_ntor = max_ntor
        self.mpol_shape = max_mpol + 1  # maximum of first fourier-modes matrix index
        self.ntor_shape = (
            max_ntor * 2 + 1
        )  # maximum of second fourier-modes matrix index
        self.poloidal_modes = torch.arange(0, self.max_mpol + 1)  # [:, None]
        self.toroidal_modes = torch.arange(-self.max_ntor, self.max_ntor + 1)
        idx = self.mpol_shape * self.ntor_shape

        #  grid for fourier features
        self.ns = ns
        self.ntheta = ntheta
        self.nzeta = nzeta
        self.nfp = nfp

        self.R_branch = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.Tanh(),
            # torch.nn.Linear(width, width),
            # torch.nn.Tanh(),
            torch.nn.Linear(width, idx),
        )
        self.l_branch = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.Tanh(),
            # torch.nn.Linear(width, width),
            # torch.nn.Tanh(),
            torch.nn.Linear(width, idx),
        )
        self.Z_branch = torch.nn.Sequential(
            torch.nn.Linear(1, width),
            torch.nn.Tanh(),
            # torch.nn.Linear(width, width),
            # torch.nn.Tanh(),
            torch.nn.Linear(width, idx),
        )

        self.Rb = Rb
        self.Zb = Zb

        #
        # zero_pad = torch.nn.ZeroPad2d((0, (self.mpol_shape-self.Rb.shape[0]),
        #                                0, (self.ntor_shape-self.Rb.shape[1])))
        # self.Rb = zero_pad(self.Rb)
        # self.Zb = zero_pad(self.Zb)

        # self.lb = torch.ones(self.mpol_shape, self.ntor_shape)
        # self.lb[0, :] = 0
        # self.lb[:, 0] = 0

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

        with torch.no_grad():
            shape = (max_mpol + 1, 2 * max_ntor + 1)
            self.R_branch[-1].bias.data.view(shape)[self.Rb != 0] = self.Rb[
                self.Rb != 0
            ]
            self.Z_branch[-1].bias.data.view(shape)[self.Zb != 0] = self.Zb[
                self.Zb != 0
            ]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: Tensor) -> Tensor:
        rho = x[:, 0].view(-1, 1)
        theta = x[:, 1].view(-1, 1)
        zeta = x[:, 2].view(-1, 1)

        # compute R, lambda, Z
        rho_factor = torch.cat(
            [rho**m for m in range(self.mpol_shape)], dim=-1
        ).unsqueeze(-1)
        # rho_factor = rho_factor.view(-1, self.ntheta, self.nzeta, rho_factor.shape[1], 1)

        # basis = get_fourier_basis(mpol=self.max_mpol - 1, ntor=int((self.max_ntor - 1) / 2),
        #                           ntheta=self.ntheta, nzeta=self.nzeta, include_endpoint=False,
        #                           num_field_period=self.nfp, dtype=rho.dtype, device=self.device)

        # costzmn0, sintzmn0 = get_fourier_basis(mpol=self.max_mpol-1, ntor=int((self.max_ntor-1)/2),
        #                            ntheta=self.ntheta, nzeta=self.nzeta,
        #                            grid=x, include_endpoint=False, num_field_period=self.nfp,
        #                            dtype=rho.dtype, device=self.device)

        # costzmn = torch.cos((self.poloidal_modes[None, None, :, None] * theta.unique()[:, None, None, None]) -
        # (self.nfp * self.toroidal_modes[None, None, None, :] * zeta.unique()[None, :, None, None]))[None, :]
        # sintzmn = torch.sin((self.poloidal_modes[None, None, :, None] * theta.unique()[:, None, None, None]) -
        # (self.nfp * self.toroidal_modes[None, None, None, :] * zeta.unique()[None, :, None, None]))[None, :]

        # costzmn = torch.cos(
        #     self.poloidal_modes[None, None, :, None] * theta[:, None, None, None] -
        #     self.nfp * self.toroidal_modes[None, None, None, :] * zeta[None, :, None,
        #                                                            None])
        # sintzmn = torch.sin(
        #     (self.poloidal_modes[None, None, :, None] * theta[:, None, None, None]) -
        #     (self.nfp * self.toroidal_modes[None, None, None, :] * zeta[None, :, None,
        #                                                            None]))

        f = (
            self.poloidal_modes[:, None] * theta[:, None]
            - (self.nfp * self.toroidal_modes * zeta).unsqueeze(1)
        )

        costzmn = torch.cos(f)
        sintzmn = torch.sin(f)

        # R = (
        #     rho_factor
        #     * self.Rb
        #     * (1 + self.R_branch(rho).reshape(-1, self.mpol_shape, self.ntor_shape))
        # )
        # R = torch.einsum("smn,smn->s", costzmn, R).contiguous()
        rmnc = rho_factor * self.R_branch(rho).reshape(
            -1, self.mpol_shape, self.ntor_shape
        )
        rmnc[:, 0, : self.max_ntor] = 0
        R = (costzmn * rmnc).sum(dim=(1, 2))
        # l = (
        #     rho_factor
        #     * self.lb
        #     * (1 + self.l_branch(rho).reshape(-1, self.mpol_shape, self.ntor_shape))
        # )
        lmns = rho_factor * self.l_branch(rho).reshape(
            -1, self.mpol_shape, self.ntor_shape
        )
        lmns[:, 0, : self.max_ntor + 1] = 0
        l = (sintzmn * lmns).sum(dim=(1, 2))
        # Z = (
        #     rho_factor
        #     * self.Zb
        #     * (1 + self.Z_branch(rho).reshape(-1, self.mpol_shape, self.ntor_shape))
        # )
        zmns = rho_factor * self.Z_branch(rho).reshape(
            -1, self.mpol_shape, self.ntor_shape
        )
        zmns[:, 0, : self.max_ntor + 1] = 0
        Z = (sintzmn * zmns).sum(dim=(1, 2))
        RlZ = torch.stack([R, l, Z], dim=-1)
        return RlZ
