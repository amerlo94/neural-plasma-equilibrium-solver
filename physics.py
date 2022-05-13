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

        #  Normalized Grad Shafranov equilibrium is not supported
        assert self.normalized == False

        #  VMEC wout file
        self.wout_path = wout_path

        #  Is a Solov'ev equilibrium?
        self.is_solovev = is_solovev

        self._axis_closure_ = None
        self._boundary_closure_ = None
        self._pde_closure_ = None

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

    @property
    def _mpol(self) -> int:
        return len(self.Rb)

    def p_fn(self, psi):
        psi_ = psi / self.psi_0
        p = 0
        for i, coef in enumerate(self.p):
            p += coef * psi_**i
        return p

    def fsq_fn(self, psi):
        psi_ = psi / self.psi_0
        fsq = 0
        for i, coef in enumerate(self.fsq):
            fsq += coef * psi_**i
        return fsq

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
            #  TODO: is it ok to have the same number of points as boundary, but effective always the same?
            #        the same can be achieved with a factor in front of the loss
            # axis = axis.expand(self.nboundary, 2)
            yield domain, boundary, axis

    def Rb_fn(self, theta):
        basis = torch.cos(torch.as_tensor([i * theta for i in range(self._mpol)]))
        return (self.Rb * basis).sum()

    def Zb_fn(self, theta):
        basis = torch.sin(torch.as_tensor([i * theta for i in range(self._mpol)]))
        return (self.Zb * basis).sum()

    def update_axis(self, axis_guess):
        #  Axis should have Za=0 by symmetry
        self._Ra = axis_guess[0]

    def eps(self, x: Tensor, psi: Tensor, reduction: Optional[str] = "mean") -> Tensor:
        assert reduction in ("mean", None)
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, retain_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, retain_graph=True)[:, 1]
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, retain_graph=True)
        fsq = self.fsq_fn(psi)
        dfsq_dpsi = grad(fsq, psi, retain_graph=True)
        R = x[:, 0]
        #  Force components
        gs = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        gs += mu0 * R**2 * dp_dpsi + 0.5 * dfsq_dpsi
        fR = -1 / (mu0 * R**2) * dpsi_dR * gs
        fZ = -1 / (mu0 * R**2) * dpsi_dZ * gs
        fsq = fR**2 + fZ**2
        #  grad-p
        gradpsq = dp_dpsi**2 * (dpsi_dR**2 + dpsi_dZ**2)
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
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, create_graph=True)
        fsq = self.fsq_fn(psi)
        dfsq_dpsi = grad(fsq, psi, create_graph=True)
        R = x[:, 0]
        residual = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        residual += mu0 * R**2 * dp_dpsi + 0.5 * dfsq_dpsi
        return (residual**2).sum()

    def _mae_pde_loss(self, x: Tensor, psi: Tensor) -> Tensor:
        dpsi_dx = grad(psi, x, create_graph=True)
        dpsi_dR = dpsi_dx[:, 0]
        dpsi_dZ = dpsi_dx[:, 1]
        dpsi2_dR2 = grad(dpsi_dR, x, retain_graph=True)[:, 0]
        dpsi2_dZ2 = grad(dpsi_dZ, x, retain_graph=True)[:, 1]
        p = self.p_fn(psi)
        dp_dpsi = grad(p, psi, retain_graph=True)
        fsq = self.fsq_fn(psi)
        dfsq_dpsi = grad(fsq, psi, retain_graph=True)
        R = x[:, 0]
        nabla_star = -1 / R * dpsi_dR + dpsi2_dR2 + dpsi2_dZ2
        denom = mu0 * R**2 * dp_dpsi + 0.5 * dfsq_dpsi
        return mae(nabla_star, -denom)

    def _boundary_closure(self, x: Tensor, psi: Tensor) -> Tensor:
        return ((psi - self.psi_0) ** 2).sum()

    def _axis_closure(self, x: Tensor, psi: Tensor) -> Tensor:
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

        return torch.cat(grid)

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
