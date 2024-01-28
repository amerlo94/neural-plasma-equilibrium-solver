import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from utils import ift, grad, mae, get_profile_from_wout, get_wout


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
            self.axis_closure = self._axis_closure_
            self.data_closure = self._data_closure_
            self.mae_pde_loss = self._mae_pde_loss_
        else:
            self.pde_closure = self._pde_closure
            self.boundary_closure = self._boundary_closure
            self.axis_closure = self._axis_closure
            self.data_closure = self._data_closure
            self.mae_pde_loss = self._mae_pde_loss

    def closure(
        self,
        x_domain: Tensor,
        psi_domain: Tensor,
        x_boundary: Tensor,
        psi_boundary: Tensor,
        x_axis: Optional[Tensor] = None,
        psi_axis: Optional[Tensor] = None,
        return_dict: Optional[bool] = False,
    ) -> Tensor:
        loss = {}
        loss["pde"] = self.pde_closure(x_domain, psi_domain)
        loss["boundary"] = self.boundary_closure(x_boundary, psi_boundary)
        loss["tot"] = loss["pde"] + loss["boundary"]
        if x_axis is not None:
            loss["axis"] = self.axis_closure(x_axis, psi_axis)
            loss["tot"] += loss["axis"]
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

    def _axis_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _boundary_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _data_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _pde_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _boundary_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _axis_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _mae_pde_loss(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    def _mae_pde_loss_(self, x: Tensor, psi: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def _mpol(self) -> int:
        return len(self.Rb)

    def p_fn(self, psi):
        p = 0
        for i, coef in enumerate(self.p):
            p += coef * psi**i
        return p

    def fsq_fn(self, psi):
        fsq = 0
        for i, coef in enumerate(self.fsq):
            fsq += coef * psi**i
        return fsq

    def iota_fn(self, psi):
        iota = 0
        for i, coef in enumerate(self.iota):
            iota += coef * psi**i
        return iota

    def Rb_fn(self, theta):
        basis = torch.cos(torch.as_tensor([i * theta for i in range(self._mpol)]))
        return (self.Rb * basis).sum()

    def Zb_fn(self, theta):
        basis = torch.sin(torch.as_tensor([i * theta for i in range(self._mpol)]))
        return (self.Zb * basis).sum()

    def update_axis(self, axis_guess):
        #  Axis should have Za=0 by symmetry
        self._Ra = axis_guess[0]


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
            yield domain, boundary, None

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
        #  Do not compute error at the boundary to avoid division by 0
        #  TODO: can we relax this?
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
        #  Do not compute error at the boundary to avoid division by 0
        #  TODO: can we relax this?
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

        x = x.detach()
        xrho = x[:, 0]
        ytheta = x[:, 1]

        ns = int(math.sqrt(x.shape[0]))

        #  Create plotting grid
        xrho = xrho.view(ns, ns)
        ytheta = ytheta.view(ns, ns)
        xx = self.R0 + xrho * torch.cos(ytheta)
        yy = xrho * torch.sin(ytheta)

        #  Detach and reshape tensors
        psi = psi.detach().view(xx.shape)

        ax.contour(xx, yy, psi, levels=10, **kwargs)
        ax.axis("equal")

        ax.set_xlabel(r"$R [m]$")
        ax.set_ylabel(r"$Z [m]$")

        return ax


class GradShafranovEquilibrium(Equilibrium):
    """
    The default case is a Solov'ev equilibrium as in the original VMEC paper.

    This repository keeps VMEC 2D equilibria under the `data` folder,
    they are taken from the DESC repository:

    https://github.com/PlasmaControl/DESC/tree/master/tests/inputs
    """

    def __init__(
        self,
        p: Tuple[float] = (0.125 / mu0, -0.125 / mu0),
        fsq: Tuple[float] = (4, -4 * 4 / 10),
        Rb: Tuple[float] = (
            3.9334e00,
            -1.0258e00,
            -6.8083e-02,
            -9.0720e-03,
            -1.4531e-03,
        ),
        Zb: Tuple[float] = (0, math.sqrt(10) / 2, 0, 0, 0),
        Ra: float = 3.9332,
        Za: float = 0.0,
        psi_0: float = 1,
        wout_path: Optional[str] = None,
        is_solovev: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        #  Pressure and current profile
        self.p = torch.as_tensor(p)
        self.fsq = torch.as_tensor(fsq)

        #  Boundary definition
        assert len(Rb) == len(Zb)
        self.Rb = torch.as_tensor(Rb)
        self.Zb = torch.as_tensor(Zb)

        #  Initial guess for the axis
        self.Ra = Ra
        self.Za = Za

        #  Running axis location
        self._Ra = Ra
        self._Za = Za

        #  Boundary condition on psi (i.e., psi_edge), the poloidal flux (chi in VMEC)
        self.psi_0 = psi_0

        #  VMEC wout file
        self.wout_path = wout_path

        #  Is a Solov'ev equilibrium?
        self.is_solovev = is_solovev

    @classmethod
    def from_vmec(cls, wout_path, **kwargs):
        """
        Instatiate Equilibrium from VMEC wout file.

        Example:

        >>> from physics import GradShafranovEquilibrium
        >>> equi = GradShafranovEquilibrium.from_vmec("data/wout_DSHAPE.nc")
        >>> equi.psi_0
        -0.665
        """

        pressure = get_profile_from_wout(wout_path, "p")
        fsq = get_profile_from_wout(wout_path, "f")

        wout = get_wout(wout_path)

        Rb = wout["rmnc"][-1].data
        Zb = wout["zmns"][-1].data

        #  Remove trailing zeros in boundary definition
        Rb = Rb[Rb != 0]
        Zb = Zb[: len(Rb)]
        Rb, Zb = map(tuple, (Rb, Zb))

        Ra = wout["raxis_cc"][:].data.item()
        Za = wout["zaxis_cs"][:].data.item()

        psi_0 = wout["chi"][-1].data.item()

        return cls(
            p=pressure,
            fsq=fsq,
            Rb=Rb,
            Zb=Zb,
            Ra=Ra,
            Za=Za,
            psi_0=psi_0,
            wout_path=wout_path,
            **kwargs,
        )

    def __iter__(self):

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        while True:
            #  Domain collocation points
            #  Create grid by scaling the boundary from the LCFS to the axis
            #  Achtung: these are not flux surfaces!
            domain = []
            ns = int(math.sqrt(self.ndomain))
            hs = torch.rand(ns, generator=generator) ** 2
            for s in hs:
                theta = (2 * torch.rand(ns, generator=generator) - 1) * math.pi
                Rb = torch.as_tensor([self.Rb_fn(t) for t in theta])
                Zb = torch.as_tensor([self.Zb_fn(t) for t in theta])
                R = (Rb - self._Ra) * s + self._Ra
                Z = (Zb - self._Za) * s + self._Za
                domain.append(torch.stack([R, Z], dim=-1))
            domain = torch.cat(domain)
            #  Boundary collocation points
            theta = (2 * torch.rand(self.nboundary, generator=generator) - 1) * math.pi
            R = torch.as_tensor([self.Rb_fn(t) for t in theta])
            Z = torch.as_tensor([self.Zb_fn(t) for t in theta])
            boundary = torch.stack([R, Z], dim=-1)
            #  Axis point
            axis = torch.Tensor([self._Ra, self._Za]).view(1, 2)
            if self.normalized:
                yield domain / self.Rb[0], boundary / self.Rb[0], axis / self.Rb[0]
            yield domain, boundary, axis

    def eps(self, x: Tensor, psi: Tensor, reduction: Optional[str] = "mean") -> Tensor:
        assert reduction in ("mean", None)
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, retain_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, retain_graph=True)[:, 1]
        #  Compute normalized poloidal flux
        psi_ = psi if self.normalized else psi / self.psi_0
        p = self.p_fn(psi_)
        dp_dpsi = grad(p, psi, retain_graph=True)
        fsq = self.fsq_fn(psi_)
        dfsq_dpsi = grad(fsq, psi, retain_graph=True)
        R = x[:, 0]
        #  Force components
        nabla_star = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        if self.normalized:
            nabla_star *= self.psi_0 / self.Rb[0] ** 2
        term = mu0 * R**2 * dp_dpsi
        if self.normalized:
            term *= self.Rb[0] ** 2 / self.psi_0
        gs = nabla_star + term
        term = 0.5 * dfsq_dpsi
        if self.normalized:
            term *= 1 / self.psi_0
        gs += term
        fR = -1 / (mu0 * R**2) * dpsi_dR * gs
        fZ = -1 / (mu0 * R**2) * dpsi_dZ * gs
        fsq = fR**2 + fZ**2
        if self.normalized:
            fsq *= self.psi_0**2 / self.Rb[0] ** 6
        #  grad-p
        gradpsq = dp_dpsi**2 * (dpsi_dR**2 + dpsi_dZ**2)
        if self.normalized:
            gradpsq *= 1 / self.Rb[0] ** 2
        if reduction is None:
            #  Compute the local normalized force balance
            return torch.sqrt(fsq / gradpsq)
        if reduction == "mean":
            #  Compute the normalized averaged force balance
            #  The `x` domain is not enforced to be on a equally spaced grid,
            #  so the sum() here is not strictly an equivalent to an integral
            return torch.sqrt(fsq.sum() / gradpsq.sum())

    def _pde_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, create_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, create_graph=True)[:, 1]
        p = self.p_fn(psi / self.psi_0)
        dp_dpsi = grad(p, psi, create_graph=True)
        fsq = self.fsq_fn(psi / self.psi_0)
        dfsq_dpsi = grad(fsq, psi, create_graph=True)
        R = x[:, 0]
        residual = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        residual += mu0 * R**2 * dp_dpsi + 0.5 * dfsq_dpsi
        return (residual**2).sum()

    def _pde_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, create_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, create_graph=True)[:, 1]
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, create_graph=True)
        fsq = self.fsq_fn(psi)
        dfsq_dpsi = grad(fsq, psi, create_graph=True)
        R = x[:, 0]
        residual = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        residual += mu0 * self.Rb[0] ** 4 / self.psi_0**2 * R**2 * dp_dpsi
        residual += 0.5 * self.Rb[0] ** 2 / self.psi_0**2 * dfsq_dpsi
        return (residual**2).sum()

    def _mae_pde_loss(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, retain_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, retain_graph=True)[:, 1]
        p = self.p_fn(psi / self.psi_0)
        dp_dpsi = grad(p, psi, retain_graph=True)
        fsq = self.fsq_fn(psi / self.psi_0)
        dfsq_dpsi = grad(fsq, psi, retain_graph=True)
        R = x[:, 0]
        nabla_star = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        denom = mu0 * R**2 * dp_dpsi + 0.5 * dfsq_dpsi
        return mae(nabla_star, -denom)

    def _mae_pde_loss_(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, create_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, create_graph=True)[:, 1]
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, create_graph=True)
        fsq = self.fsq_fn(psi)
        dfsq_dpsi = grad(fsq, psi, create_graph=True)
        R = x[:, 0]
        nabla_star = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        denom = mu0 * self.Rb[0] ** 4 / self.psi_0**2 * R**2 * dp_dpsi
        denom += 0.5 * self.Rb[0] ** 2 / self.psi_0**2 * dfsq_dpsi
        return mae(nabla_star, -denom)

    def _boundary_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - self.psi_0) ** 2).sum()

    def _boundary_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - 1) ** 2).sum()

    def _axis_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        return (psi**2).sum()

    def _axis_closure_(self, x: Tensor, psi: Tensor) -> Tensor:
        return (psi**2).sum()

    def psi(self, x: Tensor) -> Tensor:
        """
        See Bauer1978 for the nomenclature.

        Achtung: this is the analytical solution only in case of a Solov'ev equilibrium.
        """
        assert self.is_solovev == True
        R = x[:, 0]
        Z = x[:, 1]
        l2 = self.fsq[0]  # R0**2 in VMEC
        f0 = -self.fsq[1] / 4 / l2  # beta1 in VMEC
        p0 = self.p[0] * mu0  # beta0 in VMEC
        #  Get axis from R at boundary
        Ra = math.sqrt(self.Rb.sum().item() ** 2 + math.sqrt(8 / p0))
        return f0 * l2 * Z**2 + p0 / 8 * (R**2 - Ra**2) ** 2

    def grid(self, ns: int = None, normalized: bool = None) -> Tensor:

        if ns is None:
            ns = int(math.sqrt(self.ndomain))

        if normalized is None:
            normalized = self.normalized

        Rb = ift(self.Rb, basis="cos", ntheta=ns)
        Zb = ift(self.Zb, basis="sin", ntheta=ns)

        grid = []

        #  Create grid by linearly scaling the boundary from the LCFS to the axis
        #  Achtung: these are not flux surfaces!
        hs = 1 / (ns - 1)
        for i in range(ns):
            R = (Rb - self._Ra) * i * hs + self._Ra
            Z = (Zb - self._Za) * i * hs + self._Za
            grid.append(torch.stack([R, Z], dim=-1))

        grid = torch.cat(grid)

        if normalized:
            grid /= self.Rb[0]

        return grid

    def fluxplot(self, x, psi, ax, filled: Optional[bool] = False, **kwargs):

        x = x.detach()
        R = x[:, 0]
        Z = x[:, 1]

        ns = int(math.sqrt(x.shape[0]))

        #  Create plotting grid
        xx = R.view(ns, ns)
        yy = Z.view(ns, ns)

        #  Detach and reshape tensors
        psi = psi.detach().view(xx.shape)

        if filled:
            cs = ax.contourf(xx, yy, psi, **kwargs)
            ax.get_figure().colorbar(cs)
        else:
            cs = ax.contour(xx, yy, psi, levels=10, **kwargs)
            ax.clabel(cs, inline=True, fontsize=10, fmt="%1.3f")
        ax.axis("equal")

        ax.set_xlabel(r"$R [m]$")
        ax.set_ylabel(r"$Z [m]$")

        return ax

    def fluxsurfacesplot(
        self,
        x,
        ax,
        psi: Optional[Tensor] = None,
        ns: Optional[int] = None,
        nplot: Optional[int] = 10,
    ):
        """
        Plot flux surfaces on (R, Z) plane.

        TODO: improve ns and nplot handling.
        """

        assert len(x.shape) == 2

        if ns is None:
            #  Infer number of flux surfaces
            ns = int(math.sqrt(x.shape[0]))

        #  Create plotting grid
        xx = x[:, 0].view(ns, -1)
        yy = x[:, 1].view(ns, -1)

        if nplot > ns:
            nplot = ns

        #  Plot nplot + 1 since the first one is the axis
        ii = torch.linspace(0, ns - 1, nplot + 1, dtype=torch.int).tolist()
        #  If psi is given, pick equally spaced flux surfaces in terms of psi
        if psi is not None:
            psi_i = torch.linspace(0, psi[-1], nplot + 1)
            ii = []
            for p in psi_i:
                idx = torch.argmin((psi - p).abs())
                ii.append(idx)

        for i in ii:
            ax.plot(xx[i], yy[i])
            if psi is not None:
                pi_half = int(xx.shape[1] / 4)
                ax.text(xx[i][pi_half], yy[i][pi_half], f"{psi[i].item():.3f}")
        ax.axis("equal")

        return ax


class InverseGradShafranovEquilibrium(Equilibrium):
    """The default case is a DSHAPE equilibrium as in the original VMEC paper."""

    def __init__(
        self,
        p: Tuple[float] = (1.6e3, -2 * 1.6e3, 1.6e3),
        iota: Tuple[float] = (1, -0.67),
        Rb: Tuple[float] = (3.51, 1.0, 0.106),
        Zb: Tuple[float] = (0, 1.47, -0.16),
        Ra: float = 3.51,
        Za: float = 0.0,
        phi_edge: float = 1,
        wout_path: Optional[str] = None,
        is_solovev: Optional[bool] = False,
        ntheta: Optional[int] = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        #  Pressure and iota profile
        self.p = torch.as_tensor(p)
        self.iota = torch.as_tensor(iota)

        #  Boundary definition
        assert len(Rb) == len(Zb)
        self.Rb = torch.as_tensor(Rb)
        self.Zb = torch.as_tensor(Zb)

        #  Initial guess for the axis
        self.Ra = Ra
        self.Za = Za

        #  Running axis location
        self._Ra = Ra
        self._Za = Za

        #  Boundary condition on phi (i.e., phi at the VMEC LCFS)
        self.phi_edge = phi_edge

        #  VMEC wout file
        self.wout_path = wout_path

        #  Number of collocation points in the poloidal direction
        self.ntheta = ntheta

        #  Is a Solov'ev equilibrium?
        self.is_solovev = is_solovev

        #  Normalized version is not supported for now
        assert self.normalized == False
        self._pde_closure_ = None
        self._boundary_closure_ = None
        self._axis_closure_ = None

    @classmethod
    def from_vmec(cls, wout_path, **kwargs):
        """
        Instatiate Equilibrium from VMEC wout file.

        Example:

        >>> from physics import InverseGradShafranovEquilibrium
        >>> equi = InverseGradShafranovEquilibrium.from_vmec("data/wout_DSHAPE.nc")
        >>> equi.phi_edge
        1.0
        """

        wout = get_wout(wout_path)

        ns = wout["ns"][:].data.item()

        pressure = wout["am"][:].data
        pressure = pressure[pressure != 0].tolist()
        iota = wout["ai"][:].data
        iota = iota[iota != 0].tolist()

        Rb = wout["rmnc"][-1].data
        Zb = wout["zmns"][-1].data

        #  Remove trailing zeros in boundary definition
        Rb = Rb[Rb != 0]
        Zb = Zb[: len(Rb)]
        Rb, Zb = map(tuple, (Rb, Zb))

        Ra = wout["raxis_cc"][:].data.item()
        Za = wout["zaxis_cs"][:].data.item()

        phi_edge = wout["phi"][-1].data.item()

        return cls(
            p=pressure,
            iota=iota,
            Rb=Rb,
            Zb=Zb,
            Ra=Ra,
            Za=Za,
            phi_edge=phi_edge,
            wout_path=wout_path,
            ndomain=ns,
            **kwargs,
        )

    def __iter__(self):

        #  Use equally spaced grid to compute volume averaged quantities in
        #  closure functions.

        while True:
            #  Domain collocation points
            #  ndomain is the number of flux surfaces
            #  Avoid to compute loss on axis due to coordinate singularity
            ns = self.ndomain
            rho = torch.linspace(0, 1, ns + 1)[1:]
            theta = (2 * torch.linspace(0, 1, self.ntheta) - 1) * math.pi
            domain = torch.cartesian_prod(rho, theta)
            #  Boundary collocation points
            #  Use equally spaced grid
            rho = torch.ones(self.ntheta)
            boundary = torch.stack([rho, theta], dim=-1)
            yield domain, boundary, None

    def _pde_closure(self, x: Tensor, RlZ: Tensor) -> Tensor:
        #  TODO: simplify the expression to reduce torch computational graph
        R = RlZ[:, 0]
        l = RlZ[:, 1]
        Z = RlZ[:, 2]
        rho = x[:, 0]
        #  Compute the flux surface profiles
        #  self.*_fn(s), where s = rho ** 2
        p = self.p_fn(rho**2)
        iota = self.iota_fn(rho**2)
        #  Compute geometry derivatives
        dR_dx = grad(R, x, create_graph=True)
        Rs = dR_dx[:, 0]
        Ru = dR_dx[:, 1]
        dl_dx = grad(l, x, create_graph=True)
        lu = dl_dx[:, 1]
        dZ_dx = grad(Z, x, create_graph=True)
        Zs = dZ_dx[:, 0]
        Zu = dZ_dx[:, 1]
        #  Compute jacobian
        jacobian = R * (Ru * Zs - Zu * Rs)
        #  Compute the magnetic fluxes derivatives
        phis = self.phi_edge * rho / torch.pi
        chis = iota * phis
        #  Compute the contravariant magnetic field components
        bsupu = chis / jacobian
        bsupv = phis / jacobian * (1 + lu)
        #  Compute the metric tensor elements
        guu = Ru**2 + Zu**2
        gus = Ru * Rs + Zu * Zs
        gvv = R**2
        #  Compute the covariant magnetic field components
        bsubs = bsupu * gus
        bsubu = bsupu * guu
        bsubv = bsupv * gvv
        #  Compute the covariant force components,
        #  actually, mu0 * f_*
        dbsubv_dx = grad(bsubv, x, create_graph=True)
        bsubus = grad(bsubu, x, create_graph=True)[:, 0]
        bsubvs = dbsubv_dx[:, 0]
        bsubsu = grad(bsubs, x, create_graph=True)[:, 1]
        ps = grad(p, x, create_graph=True)[:, 0]
        f_rho = bsupu * bsubus + bsupv * bsubvs - bsupu * bsubsu + mu0 * ps
        bsubvu = dbsubv_dx[:, 1]
        f_theta = bsubvu * bsupv
        #  Compute the squared norm of the contravariant metric tensor
        #  grad_rho**2 == gsupss
        #  grad_theta**2 == gsupuu
        grad_rho = R**2 / jacobian**2 * (Ru**2 + Zu**2)
        grad_theta = R**2 / jacobian**2 * (Rs**2 + Zs**2)
        gsupsu = R**2 / jacobian**2 * (Rs * Ru + Zs * Zu)
        #  Compute the squared L2-norm of F
        fsq = (
            f_rho**2 * grad_rho + f_theta**2 * grad_theta + 2 * f_rho * f_theta * gsupsu
        )
        #  Compute the volume-averaged ||f||2, factors missing:
        #  1. in MKS units, there should be a 1 / mu0**2 factor
        #  2. a 4 * pi**2 / ntheta factor due to volume-averaged integration
        return (fsq * jacobian.abs()).sum()

    def eps(self, x: Tensor, RlZ: Tensor, reduction: Optional[str] = "mean") -> Tensor:
        #  TODO: include equilibrium computation in a separate method,
        #        share it with `_pde_closure`
        assert reduction in ("mean", None)
        R = RlZ[:, 0]
        l = RlZ[:, 1]
        Z = RlZ[:, 2]
        rho = x[:, 0]
        #  Compute the flux surface profiles
        #  self.*_fn(s), where s = rho ** 2
        p = self.p_fn(rho**2)
        iota = self.iota_fn(rho**2)
        #  Compute geometry derivatives
        dR_dx = grad(R, x, create_graph=True)
        Rs = dR_dx[:, 0]
        Ru = dR_dx[:, 1]
        dl_dx = grad(l, x, create_graph=True)
        lu = dl_dx[:, 1]
        dZ_dx = grad(Z, x, create_graph=True)
        Zs = dZ_dx[:, 0]
        Zu = dZ_dx[:, 1]
        #  Compute jacobian
        jacobian = R * (Ru * Zs - Zu * Rs)
        #  Compute the magnetic fluxes derivatives
        phis = self.phi_edge * rho / torch.pi
        chis = iota * phis
        #  Compute the contravariant magnetic field components
        bsupu = chis / jacobian
        bsupv = phis / jacobian * (1 + lu)
        #  Compute the metric tensor elements
        guu = Ru**2 + Zu**2
        gus = Ru * Rs + Zu * Zs
        gvv = R**2
        #  Compute the covariant magnetic field components
        bsubs = bsupu * gus
        bsubu = bsupu * guu
        bsubv = bsupv * gvv
        #  Compute the covariant force components,
        #  actually, mu0 * f_*
        dbsubv_dx = grad(bsubv, x, create_graph=True)
        bsubus = grad(bsubu, x, create_graph=True)[:, 0]
        bsubvs = dbsubv_dx[:, 0]
        bsubsu = grad(bsubs, x, create_graph=True)[:, 1]
        ps = grad(p, x, create_graph=True)[:, 0]
        f_rho = bsupu * bsubus + bsupv * bsubvs - bsupu * bsubsu + mu0 * ps
        bsubvu = dbsubv_dx[:, 1]
        f_theta = bsubvu * bsupv
        #  Compute the squared norm of the contravariant metric tensor
        grad_rho = R**2 / jacobian**2 * (Ru**2 + Zu**2)
        grad_theta = R**2 / jacobian**2 * (Rs**2 + Zs**2)
        gsupsu = R**2 / jacobian**2 * (Rs * Ru + Zs * Zu)
        #  Compute the squared L2-norm of F
        fsq = (
            f_rho**2 * grad_rho + f_theta**2 * grad_theta + 2 * f_rho * f_theta * gsupsu
        )
        gradpsq = (mu0 * ps) ** 2
        if reduction is None:
            return torch.sqrt(fsq / gradpsq)
        if reduction == "mean":
            return torch.sqrt(
                (fsq * jacobian.abs()).sum() / (gradpsq * jacobian.abs()).sum()
            )

    def _mae_pde_loss(self, x: Tensor, RlZ: Tensor) -> Tensor:
        print("MAE metric has not been implemented yet for the inverse GS equilibrium")
        return 0

    def _boundary_closure(self, x: Tensor, RlZ: Tensor) -> Tensor:
        assert torch.allclose(x[:, 0], torch.ones(x.shape[0]))
        theta = x[:, 1]
        Rb = torch.as_tensor([self.Rb_fn(t) for t in theta])
        Zb = torch.as_tensor([self.Zb_fn(t) for t in theta])
        R = RlZ[:, 0]
        Z = RlZ[:, 2]
        return ((R - Rb) ** 2).sum() + ((Z - Zb) ** 2).sum()

    def _axis_closure(self, x: Tensor, RlZ: Tensor) -> Tensor:
        raise NotImplementedError()

    def grid(self, ns: int = None, normalized: bool = None) -> Tensor:

        if ns is None:
            ns = self.ndomain

        rho = torch.linspace(0, 1, ns)
        theta = (2 * torch.linspace(0, 1, self.ntheta) - 1) * math.pi
        grid = torch.cartesian_prod(rho, theta)

        return grid

    def fluxsurfacesplot(
        self,
        x,
        ax,
        interpolation: Optional[str] = None,
        phi: Optional[torch.Tensor] = None,
        nplot: int = 10,
        scalar: Optional[torch.Tensor] = None,
        contourf_kwargs: Optional[dict] = None,
        add_phi_label: bool = False,
        **kwargs,
    ):
        """
        Plot flux surfaces on (R, Z) plane.

        TODO: improve ns and nplot handling.
        """

        assert len(x.shape) == 2
        assert interpolation in (None, "linear")

        if phi is None:
            #  Infer number of flux surfaces
            ns = int(x.shape[0] / self.ntheta)
            #  Assume flux surfaces defined on rho
            phi = torch.linspace(0, 1, ns) ** 2
        else:
            ns = phi.shape[0]
            phi = phi.detach()

        x = x.detach()

        #  Create plotting grid
        R = x[:, 0].view(ns, -1)
        Z = x[:, 1].view(ns, -1)

        if nplot > ns:
            nplot = ns

        #  Plot nplot + 1 flux surfaces equally spaced in phi
        phi_ii = torch.linspace(0, phi[-1].item(), nplot + 1)
        for i, p in enumerate(phi_ii):
            if interpolation is None:
                #  Use closest available flux surface
                idx = torch.argmin((phi - p).abs())
                R_i = R[idx]
                Z_i = Z[idx]
                phi_i = phi[idx]
            elif interpolation == "linear":
                #  Perform linear interpolation
                idx_l = torch.argmin(torch.relu(p - phi))
                idx_u = idx_l + 1
                phi_i = p
                R_i = R[idx_l]
                Z_i = Z[idx_l]
                if idx_l != len(phi) - 1:
                    R_i += (
                        (p - phi[idx_l])
                        / (phi[idx_u] - phi[idx_l])
                        * (R[idx_u] - R[idx_l])
                    )
                    Z_i += (
                        (p - phi[idx_l])
                        / (phi[idx_u] - phi[idx_l])
                        * (Z[idx_u] - Z[idx_l])
                    )
            #  Plot
            if i > 0 and "label" in kwargs:
                del kwargs["label"]
            ax.plot(R_i, Z_i, **kwargs)
            if add_phi_label:
                pi_half = int(R.shape[1] / 4)
                ax.text(R_i[pi_half], Z_i[pi_half], f"{phi_i.item():.3f}")

        if scalar is not None:
            scalar = scalar.detach().view(R.shape)
            cs = ax.contourf(R, Z, scalar, **contourf_kwargs)
            ax.get_figure().colorbar(cs)

        ax.axis("equal")
        ax.set_prop_cycle(None)

        ax.set_xlabel(r"$R [m]$")
        ax.set_ylabel(r"$Z [m]$")

        return ax
