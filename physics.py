import math
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from utils import grad, mae


mu0 = 4 * math.pi * 1e-7


class Equilibrium(IterableDataset):
    def __init__(
        self,
        ndomain: int = 2500,
        nboundary: int = 50,
        normalized: bool = False,
        seed: int = 42,
    ) -> None:

        super().__init__()

        #  Number of collocation points to use in the domain
        self.ndomain = ndomain

        #  Number of collocation points to use on the boundary
        self.nboundary = nboundary

        #  Whether to use the normalized PDE system
        self.normalized = normalized

        #  Seed to initialize the random generators
        self.seed = seed

        #  Closure functions
        if normalized:
            self.pde_closure = self._pde_closure_
            self.boundary_closure = self._boundary_closure_
            self.data_closure = self._data_closure_
            self.mae_pde_loss = self._mae_pde_loss_
        else:
            self.pde_closure = self._pde_closure
            self.boundary_closure = self._boundary_closure
            self.data_closure = self._data_closure
            self.mae_pde_loss = self._mae_pde_loss

    def closure(
        self,
        x_domain: Tensor,
        psi_domain: Tensor,
        x_boundary: Tensor,
        psi_boundary: Tensor,
        return_dict: bool = False,
    ) -> Tensor:
        loss = {}
        loss["pde"] = self.pde_closure(x_domain, psi_domain)
        loss["boundary"] = self.boundary_closure(x_boundary, psi_boundary)
        loss["tot"] = loss["pde"] + loss["boundary"]
        if return_dict:
            return loss
        return loss["tot"]

    def grid(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def fluxplot(self, *args, **kwargs):
        raise NotImplementedError

    def _data_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _pde_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _boundary_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _data_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _pde_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _boundary_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _mae_pde_loss(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _mae_pde_loss_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError


class HighBetaEquilibrium(Equilibrium):
    def __init__(
        self, a: float = 0.1, A: float = 1, C: float = 10, R0: float = 0.6, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.a = a
        self.A = A
        self.C = C
        self.R0 = R0
        self.psi_0 = -2 * A * a**2 / 8

    def __iter__(self):

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if self.normalized:
            rho_b = 1.0
        else:
            rho_b = self.a

        while True:
            #  Domain collocation points
            rho = torch.empty(self.ndomain)
            rho[0] = 0
            rho[1:] = torch.rand(self.ndomain - 1, generator=generator) * rho_b
            theta = (2 * torch.rand(self.ndomain, generator=generator) - 1) * math.pi
            domain = torch.stack([rho, theta], dim=-1)
            #  Boundary collocation points
            rho = rho_b * torch.ones(self.nboundary)
            theta = (2 * torch.rand(self.nboundary, generator=generator) - 1) * math.pi
            boundary = torch.stack([rho, theta], dim=-1)
            yield domain, boundary

    def psi(self, x: Tensor) -> Tensor:
        rho = x[:, 0]
        theta = x[:, 1]
        return (
            0.125
            * (rho**2 - self.a**2)
            * (2 * self.A + self.C * rho * torch.cos(theta))
        )

    def psi_(self, x: Tensor) -> Tensor:
        rho = x[:, 0]
        theta = x[:, 1]
        return (
            0.125
            * self.a**2
            * (rho**2 - 1)
            * (2 * self.A + self.C * rho * self.a * torch.cos(theta))
            / self.psi_0
        )

    def _data_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - self.psi(x)) ** 2).sum()

    def _data_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - self.psi_(x)) ** 2).sum()

    def _pde_closure(self, x: Tensor, psi: Tensor) -> Tensor:
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

    def _mae_pde_loss(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_drho = dpsi_dx[:, 0]
        dpsi_dtheta = dpsi_dx[:, 1]
        dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
        dpsi2_dtheta2 = grad(dpsi_dtheta, x, create_graph=True)[:, 1]
        rho = x[:, 0]
        theta = x[:, 1]
        residual = rho * dpsi_drho + rho**2 * dpsi2_drho2 + dpsi2_dtheta2
        denom = rho**2 * (self.A + self.C * rho * torch.cos(theta))
        #  TODO: avoid to compute error at the boundary to avoid division by 0
        return mae(residual[rho != self.a], denom[rho != self.a])

    def _pde_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_drho = dpsi_dx[:, 0]
        dpsi_dtheta = dpsi_dx[:, 1]
        dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
        dpsi2_dtheta2 = grad(dpsi_dtheta, x, create_graph=True)[:, 1]
        rho = x[:, 0]
        theta = x[:, 1]
        residual = rho * dpsi_drho + rho**2 * dpsi2_drho2 + dpsi2_dtheta2
        residual -= (
            self.a**2
            / self.psi_0
            * rho**2
            * (self.A + self.a * self.C * rho * torch.cos(theta))
        )
        return (residual**2).sum()

    def _mae_pde_loss_(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_drho = dpsi_dx[:, 0]
        dpsi_dtheta = dpsi_dx[:, 1]
        dpsi2_drho2 = grad(dpsi_drho, x, create_graph=True)[:, 0]
        dpsi2_dtheta2 = grad(dpsi_dtheta, x, create_graph=True)[:, 1]
        rho = x[:, 0]
        theta = x[:, 1]
        residual = rho * dpsi_drho + rho**2 * dpsi2_drho2 + dpsi2_dtheta2
        denom = (
            self.a**2
            / self.psi_0
            * rho**2
            * (self.A + self.a * self.C * rho * torch.cos(theta))
        )
        #  TODO: avoid to compute error at the boundary to avoid division by 0
        return mae(residual[rho != 1], denom[rho != 1])

    def _boundary_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        rho = x[:, 0]
        boundary = rho == self.a
        return (psi[boundary] ** 2).sum()

    def _boundary_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        rho = x[:, 0]
        boundary = rho == 1
        return (psi[boundary] ** 2).sum()

    def grid(self, ns: int = None, normalized: bool = None) -> Tensor:

        if normalized is None:
            normalized = self.normalized

        if ns is None:
            ns = int(math.sqrt(self.ndomain))

        if normalized:
            rho_b = 1.0
        else:
            rho_b = self.a

        rho = torch.linspace(0, rho_b, ns)
        theta = torch.linspace(-math.pi, math.pi, ns)

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

        ax.set_xlabel(r"$R [m]$")
        ax.set_ylabel(r"$Z [m]$")

        return ax


class GradShafranovEquilibrium(Equilibrium):
    """
    The default case is the D-shape plasma of the original VMEC paper.

    The VMEC input and output file are taken from the DESC repository:

    https://github.com/PlasmaControl/DESC/tree/master/tests/inputs
    """

    def __init__(
        self,
        pressure: Tuple[float] = (1.65e3, -1.0),
        f: Tuple[float] = (1.0, -0.67),
        Rb: Tuple[float] = (3.51, -1.0, 0.106),
        Zb: Tuple[float] = (0, 1.47, 0.16),
        phi_edge: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        #  Pressure and current profile
        self.pressure = torch.as_tensor(pressure)
        self.f = torch.as_tensor(f)

        #  Boundary definition
        assert len(Rb) == len(Zb)
        self.Rb = torch.as_tensor(Rb)
        self.Zb = torch.as_tensor(Zb)

        self.phi_edge = phi_edge

    @property
    def _mpol(self) -> int:
        return len(self.Rb)

    def p_fn(self, psi):
        return self.pressure[0] * (1 + self.pressure[1] * psi**2) ** 2

    def f_fn(self, psi):
        return self.f[0] + self.f[1] * psi**2

    def __iter__(self):

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        while True:
            #  Domain collocation points
            #  Create grid by scaling the boundary from the LCFS to the axis
            #  Achtung: these are not flux surfaces!
            domain = []
            ns = int(math.sqrt(self.ndomain))
            hs = torch.empty(ns)
            #  TODO: check if this geometric axis makes sense
            #  TODO: use ~sqrt(s) to evenly cover the area, however,
            #        by doing in this way, the axis region is not well covered
            #  Always include an "axis"
            hs[0] = 0
            hs[1:] = torch.sqrt(torch.rand(ns - 1, generator=generator))
            for s in hs:
                theta = (2 * torch.rand(ns, generator=generator) - 1) * math.pi
                Rb = torch.as_tensor([self.Rb_fn(t) for t in theta])
                Zb = torch.as_tensor([self.Zb_fn(t) for t in theta])
                R = (Rb - self.Rb[0]) * s + self.Rb[0]
                Z = (Zb - self.Zb[0]) * s + self.Zb[0]
                domain.append(torch.stack([R, Z], dim=-1))
            domain = torch.cat(domain)
            #  Boundary collocation points
            theta = (2 * torch.rand(self.nboundary, generator=generator) - 1) * math.pi
            R = torch.as_tensor([self.Rb_fn(t) for t in theta])
            Z = torch.as_tensor([self.Zb_fn(t) for t in theta])
            boundary = torch.stack([R, Z], dim=-1)
            yield domain, boundary

    def Rb_fn(self, theta):
        basis = torch.cos(torch.as_tensor([i * theta for i in range(self._mpol)]))
        return (self.Rb * basis).sum()

    def Zb_fn(self, theta):
        basis = torch.sin(torch.as_tensor([i * theta for i in range(self._mpol)]))
        return (self.Zb * basis).sum()

    def _pde_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, create_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, create_graph=True)[:, 1]
        #  TODO: check this or implement me manually since it is parametric
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, create_graph=True)
        f = self.f_fn(psi)
        df_dpsi = grad(f, psi, create_graph=True)
        R = x[:, 0]
        Z = x[:, 1]
        residual = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        residual += mu0 * R**2 * dp_dpsi + f * df_dpsi
        return (residual**2).sum()

    def _mae_pde_loss(self, x: Tensor, psi: Tensor) -> Tensor:
        #  TODO: fix me!
        return 0

    def _pde_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        #  TODO: fix me!
        pass

    def _mae_pde_loss_(self, x: Tensor, psi: Tensor) -> Tensor:
        #  TODO: fix me!
        return 0

    def _boundary_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - self.phi_edge) ** 2).sum()

    def _boundary_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        #  TODO: fix me!
        pass

    def grid(self, ns: int = None, normalized: bool = None) -> Tensor:

        if ns is None:
            ns = int(math.sqrt(self.ndomain))

        theta = torch.linspace(-math.pi, math.pi, ns)
        Rb = torch.as_tensor([self.Rb_fn(t) for t in theta])
        Zb = torch.as_tensor([self.Zb_fn(t) for t in theta])

        grid = []

        #  Create grid by linearly scaling the boundary from the LCFS to the axis
        #  Achtung: these are not flux surfaces!
        hs = 1 / (ns - 1)
        for i in range(ns):
            R = (Rb - self.Rb[0]) * i * hs + self.Rb[0]
            Z = (Zb - self.Zb[0]) * i * hs + self.Zb[0]
            grid.append(torch.stack([R, Z], dim=-1))

        return torch.cat(grid)

    def fluxplot(self, x, psi, ax, *args, **kwargs):

        R = x[:, 0]
        Z = x[:, 1]

        ns = int(math.sqrt(x.shape[0]))

        #  Create plotting grid
        xx = R.view(ns, ns)
        yy = Z.view(ns, ns)

        #  Detach and reshape tensors
        psi = psi.view(xx.shape)

        ax.contour(xx, yy, psi, levels=10, **kwargs)
        ax.axis("equal")

        ax.set_xlabel(r"$R [m]$")
        ax.set_ylabel(r"$Z [m]$")

        return ax

    def _gridplot(self, x, ax):
        """
        Debug function to visualize equilibrium grid.

        TODO: evaluate to make me a utility function
        """

        R = x[:, 0]
        Z = x[:, 1]

        ns = int(math.sqrt(x.shape[0]))

        #  Create plotting grid
        xx = R.view(ns, ns)
        yy = Z.view(ns, ns)

        for i in range(ns):
            ax.scatter(xx[i], yy[i])
        ax.axis("equal")

        return ax
