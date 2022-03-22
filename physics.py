import math
from typing import Tuple

import torch
from torch import Tensor

from utils import grad


class Equilibrium:
    def __init__(self, normalized: bool = False):

        self.normalized = normalized

        #  Set closure function
        if normalized:
            self.closure = self._closure_
        else:
            self.closure = self._closure

    def data_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError()

    def pde_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError()

    def boundary_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError()

    def data_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError()

    def pde_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError()

    def boundary_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError()

    def _closure(self, x: Tensor, psi: Tensor) -> Tensor:
        loss = {}
        loss["data"] = self.data_closure(x, psi)
        loss["pde"] = self.pde_closure(x, psi)
        loss["boundary"] = self.boundary_closure(x, psi)
        loss["tot"] = loss["pde"] + loss["boundary"]
        return loss

    def _closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        loss = {}
        loss["data"] = self.data_closure_(x, psi)
        loss["pde"] = self.pde_closure_(x, psi)
        loss["boundary"] = self.boundary_closure_(x, psi)
        loss["tot"] = loss["pde"] + loss["boundary"]
        return loss

    def closure_fn(self, *args, **kwargs) -> Tensor:
        return self.closure(*args, **kwargs)["tot"]

    def get_collocation_points(self) -> Tuple[Tensor]:
        raise NotImplementedError()

    def fluxplot(self, *args, **kwargs):
        raise NotImplementedError()


class HighBetaEquilibrium(Equilibrium):
    def __init__(
        self, a: float = 0.1, A: float = 1, C: float = 10, R0: float = 0.6, **kwargs
    ):
        super().__init__(**kwargs)

        self.a = a
        self.A = A
        self.C = C
        self.R0 = R0
        self.psi_0 = -2 * A * a**2 / 8

    def psi(self, x: Tensor) -> Tensor:
        rho = x[:, 0]
        theta = x[:, 1]
        return (
            0.125
            * (rho**2 - self.a**2)
            * (2 * self.A + self.C * rho * torch.cos(theta))
        )

    def psi_(self, x: Tensor) -> Tensor:
        """
        TODO: check me!
        """
        rho = x[:, 0]
        theta = x[:, 1]
        return (
            0.125
            * self.a**2
            * (rho**2 - 1)
            * (2 * self.A + self.C * rho * self.a * torch.cos(theta))
            / self.psi_0
        )

    def data_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - self.psi(x)) ** 2).sum()

    def data_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - self.psi_(x)) ** 2).sum()

    def pde_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_drho = dpsi_dx[:, 0]
        dpsi_dtheta = dpsi_dx[:, 1]
        dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
        dpsi2_dtheta2 = grad(dpsi_dtheta, x, create_graph=True)[:, 1]
        rho = x[:, 0]
        theta = x[:, 1]
        #  The one below is the original formulation with axis singularity
        # residual = 1 / rho * dpsi_drho + dpsi2_drho2
        # residual += 1 / rho ** 2 * dpsi2_dtheta2
        # residual -= A + C * rho * torch.cos(theta)
        residual = rho * dpsi_drho + rho**2 * dpsi2_drho2 + dpsi2_dtheta2
        residual -= rho**2 * (self.A + self.C * rho * torch.cos(theta))
        return (residual**2).sum()

    def pde_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        """
        TODO: check me, this is not correct.
        """
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_drho = dpsi_dx[:, 0]
        dpsi_dtheta = dpsi_dx[:, 1]
        dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
        dpsi2_dtheta2 = grad(dpsi_dtheta, x, create_graph=True)[:, 1]
        rho = x[:, 0]
        theta = x[:, 1]
        residual = dpsi_drho + dpsi2_drho2 + dpsi2_dtheta2
        residual -= (
            self.a**2
            / self.psi_0
            * rho**2
            * (self.A + self.a * self.C * rho * torch.cos(theta))
        )
        return (residual**2).sum()

    def boundary_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        rho = x[:, 0]
        boundary = rho == self.a
        return (psi[boundary] ** 2).sum()

    def boundary_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        rho = x[:, 0]
        boundary = rho == 1
        return (psi[boundary] ** 2).sum()

    def get_collocation_points(self, ns: int = 50, kind: str = "grid") -> Tuple[Tensor]:
        if self.normalized:
            rho_b = 1.0
        else:
            rho_b = self.a
        if kind == "grid":
            rho = torch.linspace(0, rho_b, ns)
            theta = torch.linspace(-math.pi, math.pi, ns)
        elif kind == "random":
            rho = torch.empty(ns)
            rho[1:-1] = torch.rand(ns - 2) * rho_b
            rho[0] = 0
            rho[-1] = rho_b
            theta = (2 * torch.rand(ns) - 1) * math.pi
        else:
            raise NotImplementedError("kind %s is not supported", kind)
        return torch.cartesian_prod(rho, theta)

    def fluxplot(self, x, psi, ax, *args, **kwargs):

        xrho = x[:, 0]
        ytheta = x[:, 1]

        ns = int(math.sqrt(x.shape[0]))

        #  Create plotting grid
        xrho = xrho.view(ns, ns)
        ytheta = ytheta.view(ns, ns)
        xx = self.R0 + xrho * torch.cos(ytheta)
        yy = xrho * torch.sin(ytheta)

        #  Detach and reshape tensors
        psi = psi.view(xx.shape)

        ax.contour(xx, yy, psi, levels=10, **kwargs)
        ax.axis("equal")

        return ax
